FROM python:3.12 AS builder

WORKDIR /home/
COPY obj_detection/ ./obj_detection/
COPY pyproject.toml .
COPY uv.lock .
RUN pip install uv
RUN uv sync

FROM python:3.12 AS runner

WORKDIR /home/
COPY --from=builder /home/.venv .venv
ENV VIRTUAL_ENV="/home/.venv"
ENV PATH="/home/.venv/bin:$PATH"
COPY ./datasets/ ./datasets/
COPY train_object_detection.py .
COPY start.sh .

CMD [ "bash", "start.sh" ]
