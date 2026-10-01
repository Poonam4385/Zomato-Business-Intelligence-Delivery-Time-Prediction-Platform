install:
	python -m pip install -r requirements.txt

pipeline:
	python run_pipeline.py

clean:
	python -m src.cleaning

features:
	python -m src.features

models:
	python -m src.modeling

eda:
	python -m src.eda

dashboard:
	streamlit run app/streamlit_app.py

api:
	uvicorn api.main:app --reload

test:
	pytest -q
