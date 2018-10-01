From DCLL root directory, run:
bash docker/build.sh
docker tag eneftci/dcll:dev_eneftci goofang.ss.uci.edu:5000/dcll:dev_eneftci
docker push goofang.ss.uci.edu:5000/dcll:dev_eneftci

