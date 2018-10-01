FROM pytorch/pytorch

RUN mkdir -p /srv/dcll 
COPY dcll /srv/dcll/dcll
COPY samples  /srv/dcll/samples
COPY setup.py /srv/dcll/
COPY README.md /srv/dcll/

RUN cd /srv/dcll && python3 setup.py install && cd /

WORKDIR /srv/dcll/samples
COPY docker/entrypoint.sh /
RUN ln /Results /srv/dcll/samples/Results -s
RUN ln /runs /srv/dcll/samples/runs_args -s
RUN ln /data /srv/dcll/samples/data -s

ENTRYPOINT ["/entrypoint.sh"]
#CMD cd /srv/dcll/samples && python pytorch_conv3L_dvsgestures_args.py --epochs 2
