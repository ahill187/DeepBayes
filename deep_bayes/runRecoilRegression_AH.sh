#!/bin/bash

RED='\033[0;31m'
NC='\033[0m'

# DeepML and DeepJetCore must be in the same directory
TRAINPATH=/home/ahill/DeepLearning/CMSSW_10_2_0_pre5/src/DeepML/
DEEPBAYES=/home/ahill/DeepBayes

print_help() {
    echo ""
    echo -e "${RED}Usage runRecoilRegression.sh [OPTION]... ${NC}"
    echo "Wraps up environment setting and calls to train or predict models on an input directory with ROOT trees"
    echo ""
    echo "   -r         operation to perform: convert, train"
    echo "   -m         model number to train (see Train/test_TrainData_Recoil.py) or directory with train results for predict"
    echo "   -p         apply model from this directory (required if -r predict)"
    echo "   -o         directory to hold the output [opt]"
    echo "   -c         cut to apply (train only)"
    echo "   -i         directory with input trees or txt file with trees [opt]"
    echo "   -t         use this tag to filter input files [opt]"
    echo "   -w         working directory for results (default is current working directory)"
    echo ""
}

setup_env() {
    cd ${TRAINPATH}
    DEEPJETCORE=${TRAINPATH}/../DeepJetCore
    export PYTHONPATH="${TRAINPATH}/Train:${TRAINPATH}/modules:${DEEPJETCORE}/../:${PYTHONPATH}"

    #setup environment
    if [[ ${TRAINPATH} = *"CMSSW"* ]]; then
        echo "Setting up a CMSSW-based environment"
        eval `scram r -sh`
    else
        echo "Setting up a standalone-based environment"
        export PATH=/afs/cern.ch/user/p/psilva/work/Wmass/train/miniconda/bin:$PATH
        source activate deepjetLinux3
        export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so
    fi

    export LD_LIBRARY_PATH=${DEEPJETCORE}/compiled:$LD_LIBRARY_PATH
    export PATH=${DEEPJETCORE}/bin:$PATH

    ulimit -s 65532
}

run() {

    operation=$1
    rundir=$2/${operation}
    model=$3
    cut=$4
    file_list=$5
    modeldir=$6
    echo $2
    echo $operation
    echo $rundir
    mkdir -p ${rundir}

    #prepare model
    varList="tkmet_logpt,ntnpv_logpt,npvmet_logpt,tkmet_phi,tkmet_n"
    varList="${varList},tkmet_sphericity,ntnpv_sphericity,npvmet_sphericity"
    varList="${varList},absdphi_ntnpv_tk,dphi_puppi_tk"
    varList="${varList},rho,nvert,mindz,vz"
    varList="${varList},nJets"
    if [[ "${model}"   -ge "0" && "${model}" -le "9" ]]; then
        classargs="--target lne1 --varList ${varList} --regress mu"
    elif [[ "${model}" -ge "10" && "${model}" -le "19" ]]; then
        classargs="--target lne1 --varList ${varList} --regress mu,qm,qp"
    elif [[ "${model}"   -ge "50" && "${model}" -le "97" ]]; then
        classargs="--target lne1 --varList ${varList} --regress mu,sigma,a1,a2"
        if [[ "${model}" == "52" ]]; then
            classargs="--target lne1 --varList ${varList} --regress mu,sigma"
        fi
    elif [[ "${model}" -ge "100" && "${model}" -le "149" ]]; then
        toRegress="mu_e2"
        if [[ "${model}" -ge "110" && "${model}" -le "119" ]]; then
            toRegress="mu_e2,qm_e2,qp_e2"
        fi
        classargs="--target e2 --varList ${varList} --regress ${toRegress}"
    elif [[ "${model}" -ge "150" && "${model}" -le "199" ]]; then
        classargs="--target e2 --varList ${varList} --regress mu_e2,sigma_e2,a1_e2,a2_e2,n_e2"
    elif [[ "${model}" -ge "200" && "${model}" -le "249" ]]; then
        classargs="--target trueh --varList ${varList} --regress mu,sigma"
    else
        echo "Unknown model ${model}...."
        return 0
    fi

    #add the selection cut if valid
    if [[ "${cut}" != "none" ]]; then
        classargs="${classargs} --sel ${cut}";
    fi

    if [[ "${operation}" == "convert" ]]; then
        echo -e "Preparing data to train for model ${RED} ${model} ${NC} with arguments ${RED} ${classargs} ${NC}"
        trainDataDir=${rundir}/data
        rm -rf ${trainDataDir}
        convertFromRoot.py -i ${file_list} -o ${trainDataDir} -c TrainData_Recoil --noRelativePaths --classArgs "${classargs}"

        echo "Converted Root trees"

        trainDir=${rundir}/model
        rm -rf ${trainDir}
        echo $trainDataDir/dataCollection.dc > ${rundir}/treefiles.txt
        echo $trainDir >> ${rundir}/treefiles.txt
        echo $model >> ${rundir}/treefiles.txt

        #python Train/test_TrainData_Recoil_AH.py ${trainDataDir}/dataCollection.dc ${trainDir} --modelMethod ${model}
    fi

    if [[ "${operation}" == "train" ]]; then

      trainDir=${rundir}/model
      rm -rf ${trainDir}
      trainDataDir=${rundir}/data
      python ${DEEPBAYES}/deep_bayes/model.py ${trainDataDir}/dataCollection.dc ${trainDir} ${rundir}/plots --modelMethod ${model}

    fi


    if [[ -z ${modeldir} ]]; then
        modeldir=${rundir}
    fi
}

#parse command line
while getopts "h?r:m:o:c:i:t:p:w:" opt; do
    case "$opt" in
    h|\?)
        print_help
        exit 0
        ;;
    r) operation=$OPTARG
        ;;
    m) model=$OPTARG
        ;;
    p) modeldir=$OPTARG
        ;;
    o) output=$OPTARG
        ;;
    c) cut=$OPTARG
        ;;
    i) input=$OPTARG
        ;;
    t) tag=$OPTARG
        ;;
    w) work=$OPTARG
        ;;
    esac
done
if [[ -z $operation ]] || [[ -z $model ]] || [[ -z $input ]]; then
    print_help
    exit 0
fi
if [[ "${operation}" == "predict" && -z ${modeldir} ]]; then
    print_help
    exit 0
fi

#start working directory
if [[ -z $work ]]; then
    wd = pwd
    echo "Setting working directory to ${wd}"
    work=`pwd`
fi

work=${work}/regress_results
mkdir -p ${work}

echo -e "Will run ${RED} ${operation} ${NC} in ${work} and copy to ${output}"

#build list of files to use
if [ -f ${input} ]; then
    file_list=${input}
else
    file_list=${work}/file_list.txt
    rm ${file_list}
    a=(`ls ${input}`)
    for i in ${a[@]}; do
        if [[ -n ${tag} ]]; then
            if [[ $i != *"${tag}"* ]]; then
                continue
            fi
        fi
        echo ${input}/${i} >> ${file_list}
    done
fi
echo "# input files found in ${input} : `cat ${file_list} | wc -l`"

#setup environment
setup_env

#run the required operation
if [[ -z $cut ]]; then
    cut="none";
fi
run ${operation} ${work} ${model} ${cut} ${file_list} ${modeldir}


#prepare output directories
if [ -n "${output}" -a "${output}" != "${work}" ]; then
    mkdir -p ${output}
    echo -e "Moving the results to ${RED} ${output} ${NC}"

    if [[ -d "${work}/${operation}" ]]; then
        cp -rv ${work}/${operation} ${output};

        #update file association to new location
        sed -i.bak s@${work}/${operation}/predict@${output}/${operation}/predict@g ${output}/${operation}/predict/tree_association.txt
    fi

    echo -e "Train results have been copied to ${output}"
fi
