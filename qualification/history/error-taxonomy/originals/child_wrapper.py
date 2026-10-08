"""Write the true command wait status independently of its storage supervisor."""
import json,sys,subprocess,time,os
from pathlib import Path
p=Path(sys.argv[1]);start=time.monotonic();command=json.loads((p/'command.json').read_text())
child=subprocess.Popen(command)
(p/'command-pid.json').write_text(json.dumps({'pid':child.pid,'command':command})+'\n')
rc=child.wait()
f=p/'command-exit.json';tmp=p/'command-exit.json.tmp';tmp.write_text(json.dumps({'command':command,'direct_command_exit_code':rc,'seconds':time.monotonic()-start,'independent_child_wrapper_pid':os.getpid()},indent=2)+'\n');tmp.replace(f)
raise SystemExit(rc)
