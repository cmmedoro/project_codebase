
URLS = {
    "tokyo_xs": "https://drive.google.com/file/d/1nPoLSwIvrDHExJmm3NPNQ7UPN-IOA9zs/view?usp=share_link",
    "sf_xs": "https://drive.google.com/file/d/1jlEjhhvp4XRA0V4hC_0mgaJNhXUegx9i/view?usp=share_link",
    "gsv_xs": "https://drive.google.com/file/d/1FHSMvaU-AnKeg189N_PefkCURPnS57Vl/view?usp=share_link"

    #"tokyo_xs": "https://drive.google.com/file/d/1ZPc6Ndr2IArW6ki1t8DVMcCXL1gcGMd1/view?usp=share_link",
    #"sf_xs": "https://drive.google.com/file/d/1stx09_4GdIQKP-2hMeEBcU5s1IOee2Hi/view?usp=share_link",
    #"gsv_xs": "https://drive.google.com/file/d/1ZhZj6MCHnZwAd1Q_Lb8L-QRQs8g4P7Mw/view?usp=share_link"
}

import os
import gdown
import shutil

os.makedirs("data", exist_ok=True)
for dataset_name, url in URLS.items():
    print(f"Downloading {dataset_name}")
    zip_filepath = f"data/{dataset_name}.zip"
    gdown.download(url, zip_filepath, fuzzy=True)
    shutil.unpack_archive(zip_filepath, extract_dir="data")
    os.remove(zip_filepath)


#"tokyo_xs": "https://drive.google.com/file/d/15QB3VNKj93027UAQWv7pzFQO1JDCdZj2/view?usp=share_link",
#"sf_xs": "https://drive.google.com/file/d/1tQqEyt3go3vMh4fj_LZrRcahoTbzzH-y/view?usp=share_link",
#"gsv_xs": "https://drive.google.com/file/d/1q7usSe9_5xV5zTfN-1In4DlmF5ReyU_A/view?usp=share_link"


