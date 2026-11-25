
import re
import json
import logging

from src.mongodb_service import medicines_collection


logger = logging.getLogger(__name__)


def normalize_text(value):
    if not isinstance(value, str):
        return ""

    return re.sub(
        r"\s+",
        " ",
        value.strip().lower()
    )


def extract_strength(value):
    """
    Convert a strength into milligrams.

    Examples:
        500 milligrams -> 500 mg
        500mg          -> 500 mg
        0.5 g          -> 500 mg
        125mg/5ml      -> 125 mg

    Note: Concentrations and total doses are
    not clinically interchangeable.
    """

    text = normalize_text(value)

    pattern = (
        r"(?<![\w.])"
        r"(\d+(?:\.\d+)?)"
        r"\s*"
        r"(milligrams?|mg|grams?|g|"
        r"micrograms?|mcg|µg)"
        r"\b"
    )

    match = re.search(pattern, text)

    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2)

    if unit in ("grams", "gram", "g"):
        amount *= 1000

    elif unit in (
        "micrograms",
        "microgram",
        "mcg",
        "µg"
    ):
        amount /= 1000

    return round(amount, 6)


def extract_ingredients(composition):
    """
    Parse comma-separated catalogue ingredients.

    Example:
        Paracetamol(500mg),
        Serratiopeptidase(15mg)

    Returns:
        [
            {"name": "paracetamol", "strength": 500},
            {"name": "serratiopeptidase", "strength": 15}
        ]

    This parser is intended for the observed
    catalogue format, not every possible
    pharmaceutical composition format.
    """

    ingredients = []

    if not isinstance(composition, str):
        return ingredients

    for part in composition.split(","):

        part = part.strip()

        if not part:
            continue

        match = re.match(
            r"^\s*(.+?)\s*\(([^()]*)\)",
            part
        )

        if match:
            name = normalize_text(match.group(1))
            strength = extract_strength(
                match.group(2)
            )

        else:
            name = normalize_text(part)
            strength = None

        if name:
            ingredients.append({
                "name": name,
                "strength": strength
            })

    return ingredients


def get_extracted_medicines(medical_data):
    """
    Collect medicine names and dosages
    from medical extraction.
    """

    medicines = {}

    for name in medical_data.get(
        "medicine_names", []
    ):

        if not isinstance(name, str):
            continue

        name = name.strip()

        if not name:
            continue

        medicines[normalize_text(name)] = {
            "medicine_name": name,
            "dosage": ""
        }

    for prescription in medical_data.get(
        "prescription", []
    ):

        if not isinstance(prescription, dict):
            continue

        name = prescription.get(
            "medicine_name", ""
        )

        dosage = prescription.get(
            "dosage", ""
        )

        if not isinstance(name, str):
            continue

        name = name.strip()

        if not name:
            continue

        key = normalize_text(name)

        if key not in medicines:
            medicines[key] = {
                "medicine_name": name,
                "dosage": ""
            }

        if isinstance(dosage, str):
            medicines[key]["dosage"] = dosage.strip()

    return list(medicines.values())


def classify_medicine_match(
    medicine_name,
    dosage,
    product
):
    """
    Classify a catalogue candidate.

    A combination product must not be
    classified as a single-ingredient match.
    """

    expected_name = normalize_text(
        medicine_name
    )

    expected_strength = extract_strength(
        dosage
    )

    ingredients = extract_ingredients(
        product.get("composition", "")
    )

    matching_ingredients = [
        ingredient
        for ingredient in ingredients
        if ingredient["name"] == expected_name
    ]

    if not matching_ingredients:
        return None

    matched_ingredient = matching_ingredients[0]

    catalogue_strength = matched_ingredient[
        "strength"
    ]

    strength_matches = (
        expected_strength is not None
        and catalogue_strength is not None
        and expected_strength == catalogue_strength
    )

    is_combination = len(ingredients) > 1

    if strength_matches and not is_combination:
        match_type = (
            "single_ingredient_strength_match"
        )
        priority = 3

    elif is_combination:
        match_type = "combination_product_match"

        priority = 2 if strength_matches else 1

    else:
        match_type = "ingredient_only"
        priority = 1

    return {
        "matchType": match_type,
        "strengthMatches": strength_matches,
        "isCombinationProduct": is_combination,
        "priority": priority
    }


def find_medicine_matches(
    medicine_name,
    dosage="",
    limit=3
):
    """
    Find and rank catalogue medicines.

    Search is limited to 100 candidates.
    Results are not exhaustive.
    """

    name = normalize_text(medicine_name)

    if len(name) < 3:
        return []

    limit = max(1, min(int(limit), 10))

    escaped_name = re.escape(name)

    query = {
        "$or": [
            {
                "composition": {
                    "$regex": escaped_name,
                    "$options": "i"
                }
            },
            {
                "productName": {
                    "$regex": escaped_name,
                    "$options": "i"
                }
            }
        ]
    }

    projection = {
        "_id": 0,
        "productID": 1,
        "productName": 1,
        "composition": 1,
        "manufacturer": 1,
        "MRP": 1,
        "packaging": 1,
        "schedule": 1
    }

    candidates = list(
        medicines_collection.find(
            query,
            projection
        ).limit(100).max_time_ms(15000)
    )

    ranked = []

    for product in candidates:

        classification = classify_medicine_match(
            medicine_name,
            dosage,
            product
        )

        if classification is None:
            continue

        ranked.append({
            "matchType": classification["matchType"],
            "strengthMatches": classification[
                "strengthMatches"
            ],
            "isCombinationProduct": classification[
                "isCombinationProduct"
            ],
            "priority": classification["priority"],
            "catalogueProduct": product
        })

    ranked.sort(
        key=lambda item: (
            -item["priority"],
            str(
                item["catalogueProduct"].get(
                    "productName", ""
                )
            )
        )
    )

    results = []

    for item in ranked[:limit]:
        item.pop("priority")
        results.append(item)

    return results


def match_extracted_medicines(medical_data):
    """
    Match extracted medicines with MongoDB.
    """

    extracted_medicines = get_extracted_medicines(
        medical_data
    )

    results = []

    for item in extracted_medicines:

        name = item["medicine_name"]
        dosage = item["dosage"]

        matches = find_medicine_matches(
            name,
            dosage,
            limit=3
        )

        results.append({
            "extractedMedicine": name,
            "extractedDosage": dosage,
            "matchFound": len(matches) > 0,
            "matchCountReturned": len(matches),
            "catalogueMatches": matches
        })

    return results


if __name__ == "__main__":

    sample_medical_data = {
        "medicine_names": ["Paracetamol"],
        "prescription": [
            {
                "medicine_name": "Paracetamol",
                "dosage": "500 milligrams"
            }
        ]
    }

    result = match_extracted_medicines(
        sample_medical_data
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
            ensure_ascii=False
        )
    )
