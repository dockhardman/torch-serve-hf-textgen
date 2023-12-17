FROM pytorch/torchserve:latest-gpu

ENV MODEL_NAME=llama2-7b-chat

# Copy model artifacts, custom handler and other dependencies
COPY ./requirement.txt /home/model-server/
COPY ./model_store/$MODEL_NAME.mar /home/model-server/model_store/$MODEL_NAME.mar
COPY ./config/config.properties /home/model-server/config/config.properties

# Install Dependencies
RUN pip install -r /home/model-server/requirement.txt

EXPOSE 8085
EXPOSE 8086

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
    "--start", \
    "--ts-config=/home/model-server/config/config.properties", \
    "--models", \
    "$MODEL_NAME=$MODEL_NAME.mar", \
    "--model-store", \
    "/home/model-server/model_store"]
