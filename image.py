from simple_image_download import simple_image_download as sid
response=sid.simple_image_download
keywords=["road crack"]
for kw in keywords:
    response().download(kw,500)