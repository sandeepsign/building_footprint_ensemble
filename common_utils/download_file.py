# Source: https://stackoverflow.com/a/39103880
import os
from os.path import basename
import urllib.request
from urllib.parse import urlsplit

def url2name(url):
    return basename(urlsplit(url)[2])

def download_file(url, out_path, prefix_filename):
    local_name = url2name(url)
    req = urllib.request.Request(url)
    r = urllib.request.urlopen(req)
    if r.info().get('Content-Disposition'):
        # If the response has Content-Disposition, we take file name from it
        local_name = r.info()['Content-Disposition'].split('filename=')[1]
        if local_name[0] == '"' or local_name[0] == "'":
            local_name = local_name[1:-1]
    elif r.url != url:
        # if we were redirected, the real file name we take from the final URL
        local_name = url2name(r.url)

    full_local_name = os.path.join(out_path, prefix_filename + local_name)
    f = open(full_local_name, 'wb')
    f.write(r.read())
    f.close()

    return local_name, full_local_name