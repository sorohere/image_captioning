cd deploy/model

wget "https://github.com/sorohere/flickr-dataset/releases/download/v0.2.0/model.zip"
wget "https://github.com/sorohere/flickr-dataset/releases/download/v0.2.0/vocab.zip"

unzip model.zip 
unzip vocab.zip 

rm model.zip
rm vocab.zip

rm -rf __MACOSX

cd ..
cd ..