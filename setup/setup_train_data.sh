cd $DATA_DIR
repo_name=sft_data
repo_path=https://huggingface.co/datasets/Share4oReasoning/$repo_name

# ----------------------------
# check lfs is installed
if ! [ -x "$(command -v git-lfs)" ]; then
    echo "Error: git-lfs is not installed." >&2
    echo "Please install git-lfs first:
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs"
    exit 1
fi
git clone $repo_path

# ----------------------------
target_dir=$IMAGE_DATA_DIR
cd $repo_name/image_data
# loop all tar.gz and unzip
for file in *.tar.gz; do
    echo "unzip $file"
    (
        tar -xzf "$file" -C $target_dir
        echo "$file is finished."
    ) &
done

wait

# data mix from pretrain
cd $repo_name/image_data/image_mix
for file in *.tar.gz; do
    echo "unzip $file"
    (
        tar -xzf "$file" -C $target_dir
        echo "$file is finished."
    ) &
done
wait

# ----------------------------
cd $CODE_DIR