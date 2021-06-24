FROM python:3.7-stretch
WORKDIR /deploy/
COPY ./requirements.txt /deploy/
COPY ./Ischm_App.py /deploy/
COPY ./score_objects.pkl /deploy/
COPY ./helper.py /deploy/
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "Ischm_App.py"]