import os
import os.path as osp
import errno

def args_to_string(args):
    output = 'Arguments:\n'
    d = vars(args)
    for k,v in d.items():
        if (isinstance(v, str)):
            tmp = f'--{k} \'{v}\' \\\n'
        elif (isinstance(v, bool)):
            if v is True:
                tmp = f'--{k} \\\n'
            else:
                tmp = ''
        else:
            tmp = f'--{k} {v} \\\n'
        output += tmp
    return output

def str_arg_to_int_list(str_arg):
    if str_arg is None:
        return None
    else:
        res = []
        for item in str_arg.split(','):
            if item == 'None':
                res.append(None)
            else:
                res.append(int(item))
        return res
    
def str_arg_to_bool_list(str_arg):
    if str_arg is None:
        return None
    else:
        res = []
        for item in str_arg.split(','):
            if item == 'True':
                res.append(True)
            elif item == 'False':
                res.append(False)
            else:
                raise ValueError('Argument must be either True or False')
        return res
    
def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e