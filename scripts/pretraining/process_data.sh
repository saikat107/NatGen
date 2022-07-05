CODE_HOME_DIR=`realpath ../../`;
DATA_DIR="${CODE_HOME_DIR}/data/pretraining";

function prompt() {
    echo;
    echo "Syntax: bash process_data.sh <CONFIG_NAME> <OUTPUT_DIR_NAME> [<LANGUAGES...>]";
    echo "CONFIG_NAME should be just the name of the config. The json file should be in configs/pretraining/data_config' directory."
    echo "OUTPUT_DIR_NAME should only be a string with output name";
    echo "LANGUAGES should be all languages space separated. If none given all the languages will be used!"
    exit;
}

while getopts ":h" option; do
    case $option in
        h) # display help
          prompt;
    esac
done

#if [[ $# < 2 ]]; then
#  prompt;
#fi

CONFIG_NAME="${1:-"data_processing_config"}";
OUTPUT_DIR_NAME="${2:-"processed"}";
shift;
shift;
LANGUAGES=$@;
if [[ $LANGUAGES = "" ]]; then
    echo "************************ CAUTION *************************"
    echo "No languages provided, using all the languages by default!"
    LANGUAGES="c c_sharp go java javascript php python ruby"
#    echo "=========================================================="
#    read -p "Want to use NL? [Y/N]   " choice;
#    if [[ $choice = "y" ]] || [[ $choice = "Y" ]]; then
#        LANGUAGES=$LANGUAGES" nl"
#    fi
fi
CONFIG_FILE=${CODE_HOME_DIR}/configs/pretraining/data_config/${CONFIG_NAME}.json;
output_dir="${DATA_DIR}/${OUTPUT_DIR_NAME}";
printf "Data preprocessing using $CONFIG_FILE\n"
printf "Config: "
echo `less $CONFIG_FILE`;
printf "Saving Output to : ${output_dir}\n";
printf "Languages processing: \n";
for l in $LANGUAGES; do
  printf "\t\t- $l\n";
done

function download_csnet() {
  cdir=`pwd`;
  raw_files_dir="${DATA_DIR}/raw";
  mkdir -p ${raw_files_dir};
  cd ${raw_files_dir};
  for lang in python java go php javascript ruby; do
    FILE="${lang}.pkl";
    if [[ ! -f "$FILE" ]]; then
      wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/${lang}.zip;
      unzip ${lang}.zip;
      rm -rf ${lang};
      rm ${lang}.zip;
      rm ${lang}_licenses.pkl;
      mv ${lang}_dedupe_definitions_v2.pkl $FILE;
    else
      echo "$FILE already exists";
    fi
  done
  cd $cdir;
}

function download_c_and_cs() {
  cdir=`pwd`;
  raw_files_dir="${DATA_DIR}/raw";
  mkdir -p ${raw_files_dir};
  cd ${raw_files_dir};
  c_file="c.pkl"
  if [[ ! -f "$c_file" ]]; then
    fileid="1wZUpQgPhpK7bt8vz8xXEJDvo1BL97_Px"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' \
        ./cookie`&id=${fileid}" -o ${c_file}
    rm ./cookie
  else
    echo "$c_file already exists";
  fi

  cs_file="c_sharp.pkl"
  if [[ ! -f "$cs_file" ]]; then
    fileid="1kaYgfgR7cPPO7HkGQraq8j_g8LezRJKx"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' \
        ./cookie`&id=${fileid}" -o ${cs_file}
    rm ./cookie
  else
    echo "$cs_file already exists";
  fi

#  nl_file="nl.pkl"
#  if [[ ! -f "$nl_file" ]]; then
#    fileid="1wLkvQDZsMt82ZW4u_Oe8Sg9QkHcsG9nY"
#    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
#    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' \
#        ./cookie`&id=${fileid}" -o ${nl_file}
#    rm ./cookie
#  else
#    echo "$nl_file already exists";
#  fi
  cd $cdir;
}

function process() {
  export PYTHONPATH=$PYTHONPATH:$CODE_HOME_DIR;
  input_dir="${DATA_DIR}/raw";
  python ${CODE_HOME_DIR}/src/pretraining/prepare_data.py \
    --input_dir ${input_dir} \
    --output_dir ${output_dir} \
    --langs ${LANGUAGES} \
    --parser_path ${CODE_HOME_DIR}/parser/languages.so \
    --workers 8 \
    --processing_config_file ${CONFIG_FILE};
}

download_csnet;
download_c_and_cs;
process;

