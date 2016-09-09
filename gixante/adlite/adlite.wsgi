[uwsgi]
wsgi-file = adlite.py
callable = app

master = true
processes = 2

socket = adlite.sock
chmod-socket = 660
vacuum = true

die-on-term = true

