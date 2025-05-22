rm -rf dist/ build/ && python3 -m build && twine upload --repository pypi dist/*
