#!/usr/bin/env python3
"""
任务队列管理脚本
用于将failed和processing目录中的.task文件移回pending目录
"""

import paramiko
import logging
import sys
from datetime import datetime
import os
from pathlib import Path

# 配置日志
def setup_logging():
    """设置日志配置"""
    log_filename = f"task_queue_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 创建logger
    logger = logging.getLogger('TaskQueueManager')
    logger.setLevel(logging.DEBUG)
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TaskQueueManager:
    def __init__(self, hostname, username, password=None, key_filename=None, port=22):
        """
        初始化任务队列管理器
        
        Args:
            hostname: SSH服务器地址
            username: SSH用户名
            password: SSH密码（可选）
            key_filename: SSH私钥文件路径（可选）
            port: SSH端口（默认22）
        """
        self.hostname = hostname
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.port = port
        self.logger = logging.getLogger('TaskQueueManager')
        self.ssh = None
        self.sftp = None
        
        # 基础路径
        self.base_path = "/tank/zf281/task_queue/representation_inference"
        
    def connect(self):
        """建立SSH连接"""
        try:
            self.logger.info(f"正在连接到 {self.username}@{self.hostname}:{self.port}")
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 连接参数
            connect_kwargs = {
                'hostname': self.hostname,
                'port': self.port,
                'username': self.username,
            }
            
            # 使用密码或密钥
            if self.password:
                connect_kwargs['password'] = self.password
            elif self.key_filename:
                connect_kwargs['key_filename'] = self.key_filename
            
            self.ssh.connect(**connect_kwargs)
            self.sftp = self.ssh.open_sftp()
            self.logger.info("SSH连接成功建立")
            return True
            
        except Exception as e:
            self.logger.error(f"SSH连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开SSH连接"""
        try:
            if self.sftp:
                self.sftp.close()
            if self.ssh:
                self.ssh.close()
            self.logger.info("SSH连接已断开")
        except Exception as e:
            self.logger.error(f"断开连接时出错: {e}")
    
    def execute_command(self, command):
        """执行SSH命令"""
        try:
            stdin, stdout, stderr = self.ssh.exec_command(command)
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            
            if err:
                self.logger.warning(f"命令警告/错误: {err}")
            
            return out, err
        except Exception as e:
            self.logger.error(f"执行命令失败 [{command}]: {e}")
            return None, str(e)
    
    def list_files(self, remote_path, pattern="*.task"):
        """列出远程目录中的文件"""
        try:
            # 使用find命令查找所有匹配的文件
            if pattern == "*.task":
                cmd = f"find {remote_path} -name '*.task' -type f 2>/dev/null"
            else:
                cmd = f"find {remote_path} -type f 2>/dev/null"
            
            out, err = self.execute_command(cmd)
            
            if out:
                files = out.split('\n')
                files = [f for f in files if f]  # 过滤空行
                self.logger.debug(f"在 {remote_path} 中找到 {len(files)} 个文件")
                return files
            return []
            
        except Exception as e:
            self.logger.error(f"列出文件失败 [{remote_path}]: {e}")
            return []
    
    def list_subdirectories(self, remote_path):
        """列出远程目录中的子目录"""
        try:
            cmd = f"find {remote_path} -maxdepth 1 -type d -not -path {remote_path} 2>/dev/null"
            out, err = self.execute_command(cmd)
            
            if out:
                dirs = out.split('\n')
                dirs = [d for d in dirs if d]  # 过滤空行
                self.logger.debug(f"在 {remote_path} 中找到 {len(dirs)} 个子目录")
                return dirs
            return []
            
        except Exception as e:
            self.logger.error(f"列出子目录失败 [{remote_path}]: {e}")
            return []
    
    def move_file(self, source, destination):
        """移动文件"""
        try:
            cmd = f"mv '{source}' '{destination}'"
            out, err = self.execute_command(cmd)
            
            if not err or "No such file or directory" not in err:
                self.logger.debug(f"文件移动成功: {source} -> {destination}")
                return True
            else:
                self.logger.error(f"文件移动失败: {source} -> {destination}: {err}")
                return False
                
        except Exception as e:
            self.logger.error(f"移动文件失败 [{source}]: {e}")
            return False
    
    def remove_directory(self, directory):
        """删除目录"""
        try:
            cmd = f"rm -rf '{directory}'"
            out, err = self.execute_command(cmd)
            
            if not err:
                self.logger.debug(f"目录删除成功: {directory}")
                return True
            else:
                self.logger.error(f"目录删除失败: {directory}: {err}")
                return False
                
        except Exception as e:
            self.logger.error(f"删除目录失败 [{directory}]: {e}")
            return False
    
    def process_failed_tasks(self):
        """处理failed目录中的任务"""
        self.logger.info("开始处理failed目录中的任务...")
        
        failed_path = os.path.join(self.base_path, "failed")
        pending_path = os.path.join(self.base_path, "pending")
        
        # 获取所有.task文件
        task_files = self.list_files(failed_path, "*.task")
        
        if not task_files:
            self.logger.info("failed目录中没有找到.task文件")
            return 0
        
        self.logger.info(f"在failed目录中找到 {len(task_files)} 个.task文件")
        
        # 移动文件
        moved_count = 0
        for task_file in task_files:
            filename = os.path.basename(task_file)
            destination = os.path.join(pending_path, filename)
            
            if self.move_file(task_file, destination):
                moved_count += 1
                self.logger.info(f"已移动: {filename}")
            else:
                self.logger.error(f"移动失败: {filename}")
        
        self.logger.info(f"从failed目录成功移动了 {moved_count}/{len(task_files)} 个文件")
        return moved_count
    
    def process_processing_tasks(self):
        """处理processing目录中的任务"""
        self.logger.info("开始处理processing目录中的任务...")
        
        processing_path = os.path.join(self.base_path, "processing")
        pending_path = os.path.join(self.base_path, "pending")
        
        # 获取所有子目录
        subdirs = self.list_subdirectories(processing_path)
        
        if not subdirs:
            self.logger.info("processing目录中没有找到子目录")
            return 0, 0
        
        self.logger.info(f"在processing目录中找到 {len(subdirs)} 个子目录")
        
        total_moved = 0
        dirs_removed = 0
        
        # 处理每个子目录
        for subdir in subdirs:
            subdir_name = os.path.basename(subdir)
            self.logger.info(f"处理子目录: {subdir_name}")
            
            # 获取子目录中的所有.task文件
            task_files = self.list_files(subdir, "*.task")
            
            if task_files:
                self.logger.info(f"  找到 {len(task_files)} 个.task文件")
                
                # 移动文件
                for task_file in task_files:
                    filename = os.path.basename(task_file)
                    destination = os.path.join(pending_path, filename)
                    
                    if self.move_file(task_file, destination):
                        total_moved += 1
                        self.logger.debug(f"  已移动: {filename}")
                    else:
                        self.logger.error(f"  移动失败: {filename}")
            else:
                self.logger.info(f"  子目录 {subdir_name} 中没有.task文件")
            
            # 删除子目录
            if self.remove_directory(subdir):
                dirs_removed += 1
                self.logger.info(f"  已删除子目录: {subdir_name}")
            else:
                self.logger.error(f"  删除子目录失败: {subdir_name}")
        
        self.logger.info(f"从processing目录成功移动了 {total_moved} 个文件，删除了 {dirs_removed}/{len(subdirs)} 个子目录")
        return total_moved, dirs_removed
    
    def run(self):
        """执行主要任务"""
        self.logger.info("="*50)
        self.logger.info("任务队列管理器开始运行")
        self.logger.info(f"目标路径: {self.base_path}")
        self.logger.info("="*50)
        
        # 建立连接
        if not self.connect():
            self.logger.error("无法建立SSH连接，程序退出")
            return False
        
        try:
            # 检查基础路径是否存在
            out, err = self.execute_command(f"test -d '{self.base_path}' && echo 'exists'")
            if out != 'exists':
                self.logger.error(f"基础路径不存在: {self.base_path}")
                return False
            
            # 处理failed目录
            self.logger.info("\n" + "-"*30)
            failed_moved = self.process_failed_tasks()
            
            # 处理processing目录
            self.logger.info("\n" + "-"*30)
            processing_moved, dirs_removed = self.process_processing_tasks()
            
            # 总结
            self.logger.info("\n" + "="*50)
            self.logger.info("任务完成总结:")
            self.logger.info(f"  - 从failed目录移动了 {failed_moved} 个文件")
            self.logger.info(f"  - 从processing目录移动了 {processing_moved} 个文件")
            self.logger.info(f"  - 删除了 {dirs_removed} 个子目录")
            self.logger.info(f"  - 总共恢复了 {failed_moved + processing_moved} 个任务到pending")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"执行过程中发生错误: {e}")
            return False
            
        finally:
            self.disconnect()

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    
    # SSH连接配置
    HOSTNAME = "otrera.caelum.ci.dev"
    USERNAME = "zf281"
    
    # 您可以选择使用密码或密钥文件
    # 选项1: 使用密码
    # PASSWORD = "your_password"
    # manager = TaskQueueManager(HOSTNAME, USERNAME, password=PASSWORD)
    
    # 选项2: 使用SSH密钥（推荐）
    # KEY_FILE = os.path.expanduser("~/.ssh/id_rsa")  # 修改为您的私钥路径
    # manager = TaskQueueManager(HOSTNAME, USERNAME, key_filename=KEY_FILE)
    
    # 选项3: 如果已配置SSH免密登录，可以尝试不提供密码和密钥
    manager = TaskQueueManager(HOSTNAME, USERNAME)
    
    # 运行任务
    success = manager.run()
    
    if success:
        logger.info("所有任务成功完成！")
    else:
        logger.error("任务执行过程中出现错误")
        sys.exit(1)

if __name__ == "__main__":
    main()