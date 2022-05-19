# import subprocess
import os
path = os.getcwd()

# def subprocess_():
#     """
#     subprocess模块执行linux命令
#     :return:
#     """
#     subprocess.call("ls") # 执行ls命令

# def system_():
#     """
#     system模块执行linux命令
#     :return:
#     """
#     # 使用system模块执行linux命令时，如果执行的命令没有返回值res的值是256
#     # 如果执行的命令有返回值且成功执行，返回值是0
#     res = os.system("ls")

def popen_(cmd):
    """
    popen模块执行linux命令。返回值是类文件对象，获取结果要采用read()或者readlines()
    :return:
    """
    res_log = os.popen(cmd).read() # 执行结果包含在val
    return res_log

def main(cmds):
    for cmd in cmds:
        res_log = popen_(cmd)
        # with open(path+"/"+cmd+"_log.txt", "w") as f:
        #     f.write(cmd)
        # f.close()
        print('res is: ', res_log)


if __name__ == '__main__':
    mode = input('input:')
    if mode == "train":
        cmds = ["rasa train"]
        main(cmds)
    elif mode == "shell":
        cmds = ["rasa shell --debug"]
        main(cmds)
