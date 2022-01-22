cd ..
# 确认你clone下来的这个文件夹叫xenon，否则视情况修改代码
mkdir -p $HOME/python_packages
rm -rf $HOME/python_packages/*
cp -r xenon $HOME/python_packages/xenon
# 默认你的shell是bash，如果是zsh，视情况修改代码
echo -e "\nexport PYTHONPATH=\$PYTHONPATH:$HOME/python_packages/xenon" >> ~/.bashrc
# 激活环境
source ~/.bashrc