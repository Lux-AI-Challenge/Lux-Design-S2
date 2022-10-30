# Dev

## Publishing to Kaggle

Fork the kaggle-environments repository. Copy the luxai2022 folder into kaggle_environments/lux_ai_2022/

For new packages not included in Kaggle's python image, clone those packages into the kaggle_environments/lux_ai_2022/ folder and add them to sys path in kaggle_environments/lux_ai_2022/lux_ai_2022.py

## Publish to PyPi

Change version number, remove previoust `dist` folder, and then run

```
python -m build
twine upload dist/* --verbose
```