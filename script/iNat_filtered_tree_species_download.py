import os
import time
import requests
import pandas as pd
from pyinaturalist import get_observations, get_taxa
from tqdm import tqdm  # Progress bar
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define species list with only scientific names
species_list = [
    "Buddleja davidii",
    "Rhus typhina",
    "Reynoutria japonica",
    "Bunias orientalis",
    "Senecio inaequidens",
    "Ailanthus altissima"
]

# Set up directories and parameters
output_dir = '/mnt/gsdata/projects/bigplantsens/5_ETH_Zurich_Citizen_Science_Segment/data/'
os.makedirs(output_dir, exist_ok=True)

# Configurable variables
images_per_species = 10000  # Number of images per species to download
num_cpus = min(os.cpu_count(), 2)  # Use up to 2 CPUs
rate_limit_delay = 1  # Delay in seconds between API requests to avoid rate limits

# If running on Linux/macOS, lock the script to the specified CPUs
if hasattr(os, 'sched_setaffinity'):
    os.sched_setaffinity(0, list(range(num_cpus)))

def get_taxon_id_for_species(species_name):
    """
    Look up the taxon ID on iNaturalist for a given species name.
    Returns the taxon ID (an integer) or None if no taxon was found.
    """
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
    """
    Download images for the given species and save them in a single folder named after the species.
    """
    species_name_sanitized = species_name.replace(" ", "_")
    species_dir = os.path.join(output_dir, species_name_sanitized)
    os.makedirs(species_dir, exist_ok=True)

    species_id = get_taxon_id_for_species(species_name)
    if species_id is None:
        print(f"❌ No taxon found for species: {species_name}")
        return

    print(f"\n📌 Processing species {idx}/{total_species}: {species_name} (Taxon ID: {species_id})")

    try:
        observations = get_observations(
            taxon_id=species_id,
            term_value_id=38,  # Green Leaves
            quality_grade='research',
            has=['photo'],
            per_page=images_per_species
        )
        time.sleep(rate_limit_delay)
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching observations for {species_name}: {e}")
        return

    if not observations['results']:
        print(f"⚠️ No observations found for {species_name} (green leaves).")
        return

    metadata_list = []
    image_count = 0

    with tqdm(total=images_per_species, desc=f"📥 Downloading {species_name}") as pbar:
        for obs in observations['results']:
            try:
                if 'photos' in obs and obs['photos']:
                    photo = obs['photos'][0]
                    url = photo['url'].replace('square', 'original')
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        file_name = f"obs_{obs['id']}_photo_{photo['id']}.jpg"
                        file_path = os.path.join(species_dir, file_name)
                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        image_count += 1
                        pbar.update(1)

                        relative_path = os.path.relpath(file_path)
                        meta = obs.copy()
                        meta['local_path'] = relative_path
                        metadata_list.append(meta)

                        if image_count >= images_per_species:
                            break
            except requests.exceptions.RequestException as e:
                print(f"❌ Failed to download image for observation {obs.get('id', 'unknown')}: {e}")

    print(f"✅ Finished downloading {image_count} images for species: {species_name}")

    if metadata_list:
        try:
            df = pd.json_normalize(metadata_list)
            csv_file = os.path.join(species_dir, "metadata.csv")
            df.to_csv(csv_file, index=False)
            print(f"✅ Metadata CSV saved for species: {species_name}")
        except Exception as e:
            print(f"❌ Failed to save metadata CSV for species {species_name}: {e}")

# Process each species in parallel
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
