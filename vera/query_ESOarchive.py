import os
import requests
from urllib.parse import urlencode
from subprocess import check_output as shell

HERE = os.path.dirname(os.path.abspath(__file__))
eso_script = os.path.join(HERE, 'eso_access_phase3.sh')


def query(star, instrument):

    url_endpoint = 'http://archive.eso.org/wdb/wdb/adp/phase3_main/query'
    user_agent = '--user-agent="ESO_PHASE3_PROGRAMMATIC_SCRIPT(Linux)"'
    # headers = {'User-Agent': 'ESO_PHASE3_PROGRAMMATIC_SCRIPT(Linux)'}
    args = dict(
        tab_dp_id='on',
        tab_object='on',
        target='',
        resolver='simbad',
        object=star,
        coord_sys='eq',
        tab_coord1=1,
        coord1='',
        tab_coord2=1,
        coord2='',
        box=r'02+09+00',
        dcoord_sys='eq',
        format='sexagesimal',
        tab_username='on',
        username='',
        tab_prog_id='on',
        prog_id='',
        tab_ins_id='on',
        ins_id=instrument,
        filter='',
        wavelength='',
        date_obs='2014-01-01..2099-DEC-31',
        tab_dataproduct_type=1,
        dataproduct_type='',
        phase3_collection='',
        tab_dataset_id=1,
        force_tabular_mode=1,
        top=2,
        wdbo='csv',
        order_main='mjd_obs desc',
    )

    full_url = url_endpoint + urlencode(args)

    file_tmp = 'output_temporary_1234.csv.tmp'
    cmd = f'wget -O {file_tmp} {user_agent} {full_url}'
    shell(cmd, shell=True)

    file = file_tmp[:-4]
    cmd = f'egrep -v "^$|^#" {file_tmp} > {file}'
    os.system(cmd)

    dataset_col_nr = shell(f"cat {file} | grep 'Dataset ID' | tr ',' '\012' | grep -n 'Dataset ID' | awk -F: '{{print $1}}'", shell=True)
    print(dataset_col_nr)
# arcfile_col_nr=`cat $resultsfullpath | grep 'Dataset ID' | tr ',' '\012' | grep -n 'ARCFILE' | awk -F: '{print $1}'`
# #filelist=`cat $resultsfullpath | grep "^[0-9]" | gawk -v dataset=$dataset_col_nr -v arcfile=$arcfile_col_nr -F, '{print "PHASE3%2B"$dataset"%2B"$arcfile}' FPAT="([^,]*)|(\"[^\"]*\")" | tr '\012' ','`
# filelist=`cat $resultsfullpath | grep "SCIENCE\." | gawk -v dataset=$dataset_col_nr -v arcfile=$arcfile_col_nr -F, '{print "PHASE3%2B"$dataset"%2B"$arcfile}' FPAT="([^,]*)|(\"[^\"]*\")" | tr '\012' ','`
# if [ "$debug" != "" ]; then
#    echo "DEBUG: $resultsfullpath"
#    echo "DEBUG: dataset_col_nr = $dataset_col_nr"
#    echo "DEBUG: arcfile_col_nr = $arcfile_col_nr"
#    echo "DEBUG: filelist={$filelist}"
# fi


    # r = requests.Request('GET', url_endpoint, params=args, headers=headers)
    # r.prepare()
    # print(r.url)
    # return r
    # resp = requests.get(url_endpoint, params=args, headers=headers)
    # print(resp.url)
    # with open('output_1234.csv', 'wb') as f:
    #     f.write(resp.content)
    # return resp



def query_ESO(star, instrument='ESPRESSO', top=None):
    args = dict(
        object=star,
        inst=instrument.upper(),
        top=top or 2,
    )
    args = [f'-{k} {v}' for k, v in args.items()]
    cmd = eso_script + ' ' + ' '.join(args)
    print(cmd)
    # os.system(cmd)
    return cmd


def download_reduced(star, instrument='ESPRESSO', ESOuser=None, top=None):
    if ESOuser is None:
        print('Must provide `ESOuser` to download ESO data')
        return
    cmd = query_ESO(star, instrument, top)
    cmd += f' -User {ESOuser}'
    os.system('bash ' + cmd)
