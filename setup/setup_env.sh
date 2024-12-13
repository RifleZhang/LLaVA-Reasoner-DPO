cd setup
echo huggingface for downloading large file
source install_hf.sh
echo install requirements
sudo pip install -r requirements.txt

# if exist set_path_secret.sh, source it
if [ -f set_path_secret.sh ]; then
    echo source set_path_secret.sh
    source set_path_secret.sh
else
    source set_path.sh
fi

cd $CODE_DIR

# install LLaVA-Reasoner-DPO
pip install -e .