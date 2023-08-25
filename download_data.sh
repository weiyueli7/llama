#!/bin/bash

drive_folder_link="https://drive.google.com/drive/folders/1pT5GgvuZkL06lyosCYKJA9Ii4uTdhMGE?usp=sharing"

mkdir -p data
cd data

data_folder="ScienceQA"

mkdir -p "$data_folder"

folder_id=$(echo "$drive_folder_link" | awk -F'/' '{print $NF}')

if ! command -v gdown &>/dev/null; then
    pip install gdown
fi

gdown --folder "$folder_id" --output "$data_folder"

for zip_file in "$data_folder"/*.zip; do
    unzip "$zip_file" -d "$data_folder"
    rm "$zip_file"
done
