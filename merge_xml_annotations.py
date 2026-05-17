# ============================================================
# merge_xml_annotations.py
# ============================================================
# Merges multiple CVAT for Images 1.1 XML annotation files
# into a single XML file.
#
# Each XML covers a different subset of images — no overlap.
# Verifies that no image name appears in more than one XML
# before merging to ensure annotations stay correctly matched.
#
# Run: python merge_xml_annotations.py
# ============================================================

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict


# ============================================================
# --- USER SETTINGS ---
# ============================================================

# Paths to the 4 input XML files
INPUT_XMLS = [
    "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/lettuce/annotations1/annotations.xml",
    "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/lettuce/annotations2/annotations.xml",
    "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/lettuce/annotations3/annotations.xml",
    "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/lettuce/annotations4/annotations.xml",
]

# Output merged XML path
OUTPUT_XML = "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/lettuce_v2/annotations/annotations.xml"

# Folder containing all images (used only for verification)
IMAGES_DIR = "/home/lexdata/Documents/LexAnnotate_demo/m_env/multi_pipeline/datasets/lettuce_v2/images"


# ============================================================
# --- Step 1: Parse all XML files ---
# ============================================================

def parse_xml(xml_path):
    """
    Parses a CVAT XML file.
    Returns:
        images      : list of <image> elements
        labels      : list of <label> elements from <meta>
        image_names : set of image name strings
    """
    assert os.path.exists(xml_path), \
        f"XML not found: {xml_path}"

    tree  = ET.parse(xml_path)
    root  = tree.getroot()

    # Extract label definitions from meta block
    labels = []
    meta   = root.find("meta")
    if meta is not None:
        task = meta.find("task")
        if task is not None:
            labels_el = task.find("labels")
            if labels_el is not None:
                labels = list(labels_el.findall("label"))

    # Extract all <image> elements
    images      = root.findall("image")
    image_names = {img.get("name", "") for img in images}

    return images, labels, image_names


# ============================================================
# --- Step 2: Check for duplicate image names ---
# ============================================================

def check_duplicates(all_parsed):
    """
    Checks that no image name appears in more than one XML.
    all_parsed: list of (xml_path, images, labels, image_names)
    Returns:
        duplicates: dict of image_name → list of xml paths
        has_dupes : bool
    """
    name_to_sources = defaultdict(list)

    for xml_path, _, _, image_names in all_parsed:
        for name in image_names:
            name_to_sources[name].append(xml_path)

    duplicates = {
        name: sources
        for name, sources in name_to_sources.items()
        if len(sources) > 1
    }
    return duplicates


# ============================================================
# --- Step 3: Merge labels (deduplicate by name) ---
# ============================================================

def merge_labels(all_labels_lists):
    """
    Merges label elements from all XMLs.
    Deduplicates by label name — same name = same label.
    Returns list of unique label elements.
    """
    seen   = {}   # name → ET.Element
    for labels in all_labels_lists:
        for lbl in labels:
            name = lbl.findtext("name", "").strip()
            if name and name not in seen:
                seen[name] = lbl

    return list(seen.values())


# ============================================================
# --- Step 4: Build merged XML ---
# ============================================================

def build_merged_xml(all_parsed, merged_labels):
    """
    Assembles the merged XML tree.
    Reassigns sequential image IDs to avoid collisions.
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # Meta block with merged labels
    meta       = ET.SubElement(root, "meta")
    task       = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "Merged Annotations"

    total_images = sum(
        len(images) for _, images, _, _ in all_parsed
    )
    ET.SubElement(task, "size").text = str(total_images)
    ET.SubElement(task, "mode").text = "annotation"

    labels_el = ET.SubElement(task, "labels")
    for lbl in merged_labels:
        labels_el.append(lbl)

    # Add all image elements with reassigned sequential IDs
    current_id = 0
    for xml_path, images, _, _ in all_parsed:
        for img_el in images:
            # Deep copy the image element
            new_img = ET.Element("image")
            # Reassign ID to avoid collisions
            new_img.set("id",     str(current_id))
            new_img.set("name",   img_el.get("name",   ""))
            new_img.set("width",  img_el.get("width",  "0"))
            new_img.set("height", img_el.get("height", "0"))

            # Copy all annotation child elements unchanged
            for child in img_el:
                new_img.append(child)

            root.append(new_img)
            current_id += 1

    return root


# ============================================================
# --- Step 5: Save prettified XML ---
# ============================================================

def save_xml(root, output_path):
    os.makedirs(os.path.dirname(output_path)
                if os.path.dirname(output_path) else ".",
                exist_ok=True)
    raw      = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(raw)
    xml_str  = reparsed.toprettyxml(indent="  ", encoding=None)

    # Remove the redundant <?xml ...?> declaration if present
    lines    = xml_str.split("\n")
    if lines[0].startswith("<?xml"):
        xml_str = "\n".join(lines[1:])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)


# ============================================================
# --- Main ---
# ============================================================

def main():
    print("\n" + "═" * 55)
    print("  CVAT XML Merger")
    print("═" * 55)

    # --------------------------------------------------------
    # Step 1 — Parse all XMLs
    # --------------------------------------------------------
    print("\n--- Step 1: Parsing XML files ---")
    all_parsed = []
    for xml_path in INPUT_XMLS:
        images, labels, image_names = parse_xml(xml_path)
        all_parsed.append((xml_path, images, labels, image_names))
        print(f"  {os.path.basename(xml_path):<25} "
              f"→ {len(images):>3} images | "
              f"{sum(len(list(img)) for img in images):>4} annotations")

    total_images = sum(len(images) for _, images, _, _ in all_parsed)
    print(f"\n  Total images across all XMLs : {total_images}")

    # --------------------------------------------------------
    # Step 2 — Check for duplicate image names
    # --------------------------------------------------------
    print("\n--- Step 2: Checking for duplicate image names ---")
    duplicates = check_duplicates(all_parsed)

    if duplicates:
        print(f"\n  DUPLICATES FOUND — merge aborted.\n")
        print(f"  {'Image Name':<40} Found In")
        print(f"  {'─'*70}")
        for name, sources in sorted(duplicates.items()):
            print(f"  {name:<40}")
            for src in sources:
                print(f"    → {os.path.basename(src)}")
        print(
            f"\n  ACTION: Rename the duplicate images in CVAT "
            f"and re-export before merging.\n"
        )
        return

    print(f"  No duplicates found — all image names are unique.")

    # --------------------------------------------------------
    # Step 3 — Verify images exist on disk (optional check)
    # --------------------------------------------------------
    print("\n--- Step 3: Verifying images exist on disk ---")
    missing = []
    for _, images, _, _ in all_parsed:
        for img_el in images:
            fname = img_el.get("name", "")
            fpath = os.path.join(IMAGES_DIR, fname)
            if not os.path.exists(fpath):
                missing.append(fname)

    if missing:
        print(f"  WARNING: {len(missing)} image(s) not found "
              f"in {IMAGES_DIR}:")
        for m in missing[:10]:
            print(f"    ✗ {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")
        print(f"  Proceeding with merge anyway.")
    else:
        print(f"  All {total_images} images verified on disk.")

    # --------------------------------------------------------
    # Step 4 — Merge labels
    # --------------------------------------------------------
    print("\n--- Step 4: Merging label definitions ---")
    merged_labels = merge_labels(
        [labels for _, _, labels, _ in all_parsed]
    )
    label_names = [
        lbl.findtext("name", "") for lbl in merged_labels
    ]
    print(f"  Unique labels ({len(label_names)}): {label_names}")

    # --------------------------------------------------------
    # Step 5 — Build and save merged XML
    # --------------------------------------------------------
    print("\n--- Step 5: Building merged XML ---")
    merged_root = build_merged_xml(all_parsed, merged_labels)

    print(f"--- Step 6: Saving to {OUTPUT_XML} ---")
    save_xml(merged_root, OUTPUT_XML)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    ann_counts = defaultdict(int)
    for _, images, _, _ in all_parsed:
        for img_el in images:
            for child in img_el:
                ann_counts[child.tag] += 1

    print(f"\n{'═'*55}")
    print(f"  Merge Complete")
    print(f"{'─'*55}")
    print(f"  Images merged     : {total_images}")
    print(f"  Unique labels     : {len(label_names)}")
    print(f"\n  Annotations by type:")
    for atype, count in sorted(ann_counts.items()):
        print(f"    {atype:<15}: {count}")
    print(f"\n  Output → {OUTPUT_XML}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
