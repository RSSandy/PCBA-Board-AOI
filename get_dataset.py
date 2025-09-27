from roboflow import Roboflow

rf = Roboflow(api_key="l5PPObaSd94UBGF7fZ2Q")
project = rf.workspace("ilick").project("pcba-zncda")
version = project.version(1)
dataset = version.download("yolov8-obb")
