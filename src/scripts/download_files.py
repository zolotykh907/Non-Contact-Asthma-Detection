import os
import requests
from tqdm import tqdm
from requests.exceptions import RequestException
import time

def download_file(url, save_path, max_retries=1, retry_delay=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(save_path, 'wb') as file, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    progress_bar.update(size)
            
            return True
        except RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Download failed.")
                return False

def get_file_size(url):
    try:
        response = requests.head(url)
        response.raise_for_status()
        return int(response.headers.get("Content-Length", 0))
    except RequestException:
        return 0

rml_folder = '/var/data/apnea/rml/'
edf_folder = '/var/data/apnea/datasets/edf/'
url_file = '/var/data/apnea/778740145531650048.txt'

os.makedirs(rml_folder, exist_ok=True)
os.makedirs(edf_folder, exist_ok=True)

with open(url_file, "r") as f:
    url_lines = f.readlines()

rml_files = os.listdir(rml_folder)
sorted_rml_files = sorted(rml_files, key=lambda x: int(x.split('-')[0]))

bad_rml_files = ['995', '1008']

for rml_file in tqdm(sorted_rml_files[3:], desc="Processing RML files"):
    patient_name = rml_file.replace('.rml', '')
    if patient_name in bad_rml_files: 
        continue

    print(f"Processing: {rml_file}")

    for line in url_lines:
        if line.endswith('.edf\n') and patient_name in line:
            url = line.strip()
            file_name = url.split('fileName=')[-1]
            save_path = os.path.join(edf_folder, file_name)

            file_size = get_file_size(url)
            
            if os.path.exists(save_path):
                if os.path.getsize(save_path) == file_size:
                    print(f'{file_name} already exists')
                    continue 
                else:
                    print(f'{file_name} is incomplete, re-downloading')

            if download_file(url, save_path):
                print(f'{file_name} downloaded successfully')
            else:
                print(f'Failed to download {file_name}')