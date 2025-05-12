cd dataset

wget "https://github.com/sorohere/flickr-dataset/releases/download/v0.1.0/flickr30k-dataset-part_00"
wget "https://github.com/sorohere/flickr-dataset/releases/download/v0.1.0/flickr30k-dataset-part_01"
wget "https://github.com/sorohere/flickr-dataset/releases/download/v0.1.0/flickr30k-dataset-part_02"

cat flickr30k-dataset-part_* > flickr30k-dataset.zip
unzip -q flickr30k-dataset.zip 

rm flickr30k-dataset-part_00 flickr30k-dataset-part_01 flickr30k-dataset-part_02
rm flickr30k-dataset.zip
rm -rf __MACOSX

echo "Downloaded Flickr30k dataset successfully."

cd ..