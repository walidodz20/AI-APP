FROM python:3.10

WORKDIR /code
COPY ./requirement.txt /code/requirement.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirement.txt

RUN useradd user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
