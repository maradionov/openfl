# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""CA api module."""

from pathlib import Path
import base64
import os
import shutil

from click import echo

pki_dir = './cert'
step_config_dir = './step_config/'
step = None
step_ca = None


def check_kill_process(pstring):
    """Kill process by name."""
    import signal

    for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)


def create_dirs():
    """Create CA directories."""
    prefix = Path('.')
    (prefix / pki_dir).mkdir(parents=True, exist_ok=True)
    (prefix / step_config_dir).mkdir(parents=True, exist_ok=True)


def download_step_bin(url, grep_name, architecture, prefix='./'):
    """
    Donwload step binaries from github.

    Args:
        url: address of latest release
        grep_name: name to grep over github assets
        architecture: architecture type to grep
        prefix: folder path to download
    """
    import requests
    import urllib.request

    result = requests.get(url)
    assets = result.json()['assets']
    urls = []
    for a in assets:
        if grep_name in a['name'] and architecture in a['name']:  # 'amd'
            urls.append(a['browser_download_url'])
    url = urls[-1]
    url = url.replace('https', 'http')
    name = url.split('/')[-1]
    print('Downloading:', name)
    urllib.request.urlretrieve(url, f'{prefix}/{name}')
    shutil.unpack_archive(f'{prefix}/{name}', f'{prefix}/step')


def init(password, ca_url):
    """
    Create certificate authority for federation.

    Args:
        password: Simple password for encrypting root private keys
        ca_url: url for ca server like: 'host:port'

    """
    global step, step_ca
    check_kill_process('step-ca')
    create_dirs()
    with open(f'{pki_dir}/pass_file', 'w') as f:
        f.write(password)

    echo('Setting Up Certificate Authority...\n')

    url = 'http://api.github.com/repos/smallstep/certificates/releases/latest'
    download_step_bin(url, 'step-ca_linux', 'amd')
    url = 'http://api.github.com/repos/smallstep/cli/releases/latest'
    download_step_bin(url, 'step_linux', 'amd')
    dirs = os.listdir('./step')
    for dir_ in dirs:
        if 'step_' in dir_:
            step = f'./step/{dir_}/bin/step'
        if 'step-ca' in dir_:
            step_ca = f'./step/{dir_}/bin/step-ca'
    assert(step and step_ca and os.path.exists(step) and os.path.exists(step_ca))
    print(step)

    echo('Create CA Config')
    os.environ["STEPPATH"] = step_config_dir
    shutil.rmtree(step_config_dir, ignore_errors=True)
    name = ca_url.split(':')[0]
    os.system(f'./{step} ca init --name name --dns {name} '
              + f'--address {ca_url}  --provisioner prov '
              + f'--password-file {pki_dir}/pass_file')

    os.system(f'./{step} ca provisioner remove prov --all')
    os.system(f'./{step} crypto jwk create {step_config_dir}/certs/pub.json '
              + f'{step_config_dir}/secrets/priv.json --password-file={pki_dir}/pass_file')
    os.system(f'./{step} ca provisioner add provisioner {step_config_dir}/certs/pub.json')
    echo('Up CA server')
    os.system(f'{step_ca} --password-file {pki_dir}/pass_file {step_config_dir}/config/ca.json  &')
    echo('\nDone.')


def get_token(name, ca_url):
    """
    Create authentication token.

    Args:
        name: common name for following certificate
                    (aggregator fqdn or collaborator name)
        ca_url: full url of CA server
    """
    print(step)
    import subprocess

    os.environ["STEPPATH"] = step_config_dir
    try:
        token = subprocess.check_output(f'./{step} ca token {name} '
                                        f'--key {step_config_dir}/secrets/priv.json '
                                        f'--password-file {pki_dir}/pass_file '
                                        f'--ca-url {ca_url} ', shell=True)
    except subprocess.CalledProcessError as exc:
        print(f'Error code {exc.returncode}: {exc.output}')
        return

    if token[-1:] == b'\n':
        token = token[:-1]
    length = len(token)
    assert(length < 10002)
    length = str(10000 + length)[-4:]
    with open(step_config_dir + '/certs/root_ca.crt', mode='rb') as file:
        root_ca = file.read()

    base64_bytes = base64.b64encode(root_ca)
    base64_message = base64_bytes.decode('ascii')
    print('Token:')
    print(length, token.decode('ascii'), base64_message, sep='')


def certify(fqdn, token_with_cert):
    """Create aggregator certificate key pair."""
    length = int(token_with_cert[:4])
    token = token_with_cert[4:length + 4]
    root_ca = token_with_cert[length + 4:]
    message_bytes = base64.b64decode(root_ca)

    # write ca cert to file
    with open(f'{pki_dir}/root_ca.crt', mode='wb') as file:
        file.write(message_bytes)
    # request to ca server
    os.system(f'./{step} ca certificate {fqdn} agg_{fqdn}.crt agg_{fqdn}.key -f --token {token}')

## Simple pipeline:
#
# from ca import *
# import time

# init('123','nnlicv674.inn.intel.com:4343')
# time.sleep(2)
# get_token('localhost', 'https://nnlicv674.inn.intel.com:4343')
# token = str(input())
# certify('localhost',token)