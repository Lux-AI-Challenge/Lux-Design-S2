# Dev

## Publishing to Kaggle

Fork the kaggle-environments repository. Copy the luxai_s2/luxai_s2 folder into kaggle_environments/luxai_s2/

For new packages not included in Kaggle's python image, clone those packages into the kaggle_environments/luxai_s2/ folder and add them to sys path in kaggle_environments/luxai_s2/luxai_s2.py

Set version = "" in version.py file as its not a "package" in the kaggle-environments repo

## Publish to PyPi

Change version number, remove previoust `dist` folder, and then run

```
python -m build
twine upload dist/* --verbose
```