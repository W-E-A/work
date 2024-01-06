# 初始化位置参数变量
params=""

# 解析命令行参数并提取value
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --*=*)
        value="${1#*=}"   # 提取值部分，例如：--key=value 中的 value
        params="$params $value" # 存储value到变量，使用空格分隔
        shift
        ;;
      *) shift ;;
    esac
  done
}

# 调用参数解析函数
parse_args "$@"

# 执行bash脚本，转发所有位置参数
bash /ai/volume/work/tools/train_v2x.sh $@