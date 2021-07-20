FROM openfaas/of-watchdog:0.7.7 as watchdog
FROM python:3.7-slim-buster

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

ARG ADDITIONAL_PACKAGE
# Alternatively use ADD https:// (which will not be cached by Docker builder)

RUN apt-get -qy update && apt-get -qy install gcc make ${ADDITIONAL_PACKAGE}

# Add non root user
RUN addgroup --system app && adduser app --system --ingroup app
RUN chown app /home/app

USER app

ENV PATH=$PATH:/home/app/.local/bin

WORKDIR /home/app/

COPY index.py           .
COPY requirements.txt   .

USER root
RUN pip install -r requirements.txt

# TPU
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get -y install curl gnupg2
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update && apt-get -y install libedgetpu1-std
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
# This repo is needed for only python3-coral-cloudiot and python3-coral-enviro
RUN echo "deb https://packages.cloud.google.com/apt coral-cloud-stable main" |  tee /etc/apt/sources.list.d/coral-cloud.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update && apt-get -y install python3-pycoral


# Build the function directory and install any user-specified components
USER app

RUN mkdir -p function
RUN touch ./function/__init__.py
WORKDIR /home/app/function/
COPY function/requirements.txt	.
RUN pip install --user -r requirements.txt

#install function code
USER root

COPY function/   .
RUN chown -R app:app ../

ARG TEST_COMMAND=tox
ARG TEST_ENABLED=false
RUN if [ "$TEST_ENABLED" == "false" ]; then \
    echo "skipping tests";\
    else \
    eval "$TEST_COMMAND"; \
    fi

WORKDIR /home/app/

#configure WSGI server and healthcheck
USER app

ENV fprocess="python index.py"

ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
