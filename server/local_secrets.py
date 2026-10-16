"""Load only optional provider credentials, without executing a shell file."""
import os
from pathlib import Path


def load_provider_credentials(root):
    path=Path(root)/'.env.local'
    if not path.is_file():return
    for line in path.read_text().splitlines():
        name,sep,value=line.strip().partition('=')
        if sep and name in ('TYPESAFE_API_KEY','JEV_API_KEY','GEMINI_API_KEY','GOOGLE_API_KEY'):
            value=value.strip()
            if len(value)>=2 and value[0]==value[-1] and value[0] in ('"',"'"):
                value=value[1:-1]
            if value:os.environ.setdefault(name,value)
