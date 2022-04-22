#! /usr/bin/python2
import xml.etree.ElementTree as ET
import sys
import base64
import os
import urllib
import urlparse
PREFIX="data:"
ATTR="{http://www.w3.org/1999/xlink}href"
DEFAULT_NS="http://www.w3.org/2000/svg"
with open(sys.argv[1]) as f:
    root = ET.parse(f)
    for e in root.findall(".//{%s}image" % DEFAULT_NS):
        href = e.get(ATTR)
        if href and href[:5]!="data:":    
            #path selection strategy:
            # 1. href if absolute
            # 2. realpath-ified href
            # 3. absref, only if the above does not point to a file
            path=os.path.realpath(href)
            if (not os.path.isfile(path)):
                if (absref != None):
                    path=absref

            try:
                path=unicode(path, "utf-8")
            except TypeError:
                path=path
                
            if (not os.path.isfile(path)):
                sys.stderr.write("No xlink:href or sodipodi:absref attributes found, or they do not point to an existing file! Unable to embed image.")
                if path:
                    sys.stderr.write(_("Sorry we could not locate %s") % str(path))
                sys.exit

            if (os.path.isfile(path)):
                file = open(path,"rb").read()
                embed=True
                if (file[:4]=='\x89PNG'):
                    type='image/png'
                elif (file[:2]=='\xff\xd8'):
                    type='image/jpeg'
                elif (file[:2]=='BM'):
                    type='image/bmp'
                elif (file[:6]=='GIF87a' or file[:6]=='GIF89a'):
                    type='image/gif'
                elif (file[:4]=='MM\x00\x2a' or file[:4]=='II\x2a\x00'):
                    type='image/tiff'
                #ico files lack any magic... therefore we check the filename instead
                elif(path.endswith('.ico')):
                    type='image/x-icon' #official IANA registered MIME is 'image/vnd.microsoft.icon' tho
                else:
                    embed=False
                if (embed):
                    print("%s\n" % e)
                    print("xlink/href=data:%s;base64,%s\n"% (type, base64.encodestring(file)))
                    e.set(ATTR, "data:%s;base64,%s\n"% (type, base64.encodestring(file)))
                else:
                    sys.stderr.write("%s is not of type image/png, image/jpeg, image/bmp, image/gif, image/tiff, or image/x-icon" % path)
                    sys.exit

root.write(sys.argv[1])
