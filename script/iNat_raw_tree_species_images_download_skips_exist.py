import os
import time
import requests
import pandas as pd
from pyinaturalist import get_observations, get_taxa
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

species_list = [
    "Abies alba",
    "Acer pseudoplatanus",
    "Betula pendula",
    "Fagus sylvatica",
    "Fraxinus excelsior",
    "Larix decidua",
    "Picea abies",
    "Pinus sylvestris",
    "Pseudotsuga menziesii",
    "Quercus rubra",
]

output_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
os.makedirs(output_dir, exist_ok=True)

images_per_species = 4000
num_cpus = min(os.cpu_count(), 32)  # Use up to 32 CPUs
rate_limit_delay = 1

if hasattr(os, 'sched_setaffinity'):
    os.sched_setaffinity(0, list(range(num_cpus)))

def get_taxon_id_for_species(species_name):
    try:
        taxon_response = get_taxa(q=species_name, rank='species')
        if taxon_response['results']:
            return taxon_response['results'][0]['id']
        else:
            return None
    except Exception as e:
        print(f"❌ Error fetching taxon for {species_name}: {e}")
        return None

def download_species_images(idx, species_name, total_species):
    species_name_sanitized = species_name.replace(" ", "_")
    species_dir = os.path.join(output_dir, species_name_sanitized)
    os.makedirs(species_dir, exist_ok=True)

    species_id = get_taxon_id_for_species(species_name)
    if species_id is None:
        print(f"❌ No taxon found for species: {species_name}")
        return

    print(f"\n📌 Processing species {idx}/{total_species}: {species_name} (Taxon ID: {species_id})")
    images_to_download = images_per_species
    per_page = 200
    page = 1
    image_count = 0
    metadata_list = []

    # Get set of already downloaded files
    existing_files = set(os.listdir(species_dir))

    with tqdm(total=images_to_download, desc=f"📥 Downloading {species_name}") as pbar:
        while image_count < images_to_download:
            try:
                observations = get_observations(
                    taxon_id=species_id,
                    without_term_value_id=19,
                    quality_grade='research',
                    has=['photo'],
                    month="1,2,5,6,7,8,9,11,12", # Observations from January-February, May through September, and November-December.
                    per_page=per_page,
                    page=page
                )
                time.sleep(rate_limit_delay)
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Error fetching observations for {species_name} on page {page}: {e}")
                break

            if not observations['results']:
                print(f"⚠️ No more observations found for {species_name} on page {page}.")
                break

            for obs in observations['results']:
                if image_count >= images_to_download:
                    break
                try:
                    if 'photos' in obs and obs['photos']:
                        photo = obs['photos'][0]
                        url = photo['url'].replace('square', 'original')
                        file_name = f"obs_{obs['id']}_photo_{photo['id']}.jpg"
                        file_path = os.path.join(species_dir, file_name)
                        if file_name in existing_files:
                            # Skip if file already exists
                            continue
                        response = requests.get(url, stream=True)
                        if response.status_code == 200:
                            with open(file_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            image_count += 1
                            pbar.update(1)
                            relative_path = os.path.relpath(file_path)
                            meta = obs.copy()
                            meta['local_path'] = relative_path
                            metadata_list.append(meta)
                except requests.exceptions.RequestException as e:
                    print(f"❌ Failed to download image for observation {obs.get('id', 'unknown')}: {e}")
            page += 1

    print(f"✅ Finished downloading {image_count} images for species: {species_name}")

    if metadata_list:
        try:
            df = pd.json_normalize(metadata_list)
            csv_file = os.path.join(species_dir, "metadata.csv")
            df.to_csv(csv_file, index=False)
            print(f"✅ Metadata CSV saved for species: {species_name}")
        except Exception as e:
            print(f"❌ Failed to save metadata CSV for species {species_name}: {e}")

with ThreadPoolExecutor(max_workers=num_cpus) as executor:
    futures = {
        executor.submit(download_species_images, idx, species_name, len(species_list)): species_name
        for idx, species_name in enumerate(species_list, start=1)
    }
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"❌ Error in parallel execution: {e}")

print("\n✅ Image download process completed successfully! 🚀")