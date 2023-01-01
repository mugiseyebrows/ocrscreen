FROM quay.io/pypa/manylinux_2_28_x86_64
WORKDIR /usr/src/app
COPY . .
CMD /bin/bash build-linux-wheels.sh
