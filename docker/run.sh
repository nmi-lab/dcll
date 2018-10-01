if [ ! -d "samples" ]; then
  echo "directory samples does not exist, exiting"
  exit
fi

docker run --rm -it --init --runtime=nvidia --ipc=host --volume=$PWD/data:/data --volume=$PWD/Results:/Results --volume=$PWD/runs:/runs -e NVIDIA_VISIBLE_DEVICES=0 eneftci/dcll:dev_eneftci
