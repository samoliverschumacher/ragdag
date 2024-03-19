import yaml

from app.main import app

spec = app.openapi()
spec_text = yaml.dump(spec)

with open("openapi/openapi.yaml", "w") as spec_write:
    spec_write.write(spec_text)
