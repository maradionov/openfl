# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Workspace module."""

from pathlib import Path
from click import Choice, Path as ClickPath
from click import group, option, pass_context
from click import echo, confirm
from subprocess import check_call
from sys import executable
from shutil import copyfile, ignore_patterns
import os

from openfl.interface.cli_helper import copytree, print_tree
from openfl.interface.cli_helper import WORKSPACE, PKI_DIR
from openfl.interface.cli_helper import SITEPACKS, OPENFL_USERDIR

step_config_dir = '/home/radionov/aggregator/step_config/'
step = './step/step_0.15.16/bin/step'
step_ca = './step/step-ca_0.15.15/bin/step-ca'


@group()
@pass_context
def workspace(context):
    """Manage Federated Learning Workspaces."""
    context.obj['group'] = 'workspace'


def create_dirs(prefix):
    """Create workspace directories."""
    echo('Creating Workspace Directories')

    (prefix / 'cert').mkdir(parents=True, exist_ok=True)  # certifications
    (prefix / 'data').mkdir(parents=True, exist_ok=True)  # training data
    (prefix / 'logs').mkdir(parents=True, exist_ok=True)  # training logs
    (prefix / 'plan').mkdir(parents=True, exist_ok=True)  # federated learning plans
    (prefix / 'save').mkdir(parents=True, exist_ok=True)  # model weight saves / initialization
    (prefix / 'src').mkdir(parents=True, exist_ok=True)  # model code

    src = WORKSPACE / 'workspace/plan/defaults'  # from default workspace
    dst = prefix / 'plan/defaults'  # to created workspace

    copytree(src=src, dst=dst, dirs_exist_ok=True)
    copyfile(WORKSPACE / 'workspace' / '.workspace', prefix / '.workspace')


def create_temp(prefix, template):
    """Create workspace templates."""
    echo('Creating Workspace Templates')

    copytree(src=WORKSPACE / template, dst=prefix, dirs_exist_ok=True,
             ignore=ignore_patterns('__pycache__'))  # from template workspace


def get_templates():
    """Grab the default templates from the distribution."""
    return [d.name for d in WORKSPACE.glob('*') if d.is_dir()
            and d.name not in ['__pycache__', 'workspace']]


@workspace.command(name='create')
@option('--prefix', required=True,
        help='Workspace name or path', type=ClickPath())
@option('--template', required=True, type=Choice(get_templates()))
def create_(prefix, template):
    """Create the workspace."""
    create(prefix, template)


def create(prefix, template):
    """Create federated learning workspace."""
    from os.path import isfile
    if not OPENFL_USERDIR.exists():
        OPENFL_USERDIR.mkdir()

    prefix = Path(prefix)
    template = Path(template)

    create_dirs(prefix)
    create_temp(prefix, template)

    requirements_filename = "requirements.txt"

    if isfile(f'{str(prefix)}/{requirements_filename}'):
        check_call([
            executable, "-m", "pip", "install", "-r",
            f"{prefix}/requirements.txt"], shell=False)
        echo(f"Successfully installed packages from {prefix}/requirements.txt.")
    else:
        echo("No additional requirements for workspace defined. Skipping...")
    prefix_hash = _get_dir_hash(str(prefix.absolute()))
    with open(OPENFL_USERDIR / f'requirements.{prefix_hash}.txt', 'w') as f:
        check_call([executable, '-m', 'pip', 'freeze'], shell=False, stdout=f)

    print_tree(prefix, level=3)


@workspace.command(name='export')
def export_():
    """Export federated learning workspace."""
    from shutil import make_archive, copytree, copy2, ignore_patterns
    from tempfile import mkdtemp
    from os import getcwd, makedirs
    from os.path import basename, join
    from plan import FreezePlan

    # TODO: Does this need to freeze all plans?
    planFile = 'plan/plan.yaml'
    try:
        FreezePlan(planFile)
    except Exception:
        echo(f'Plan file "{planFile}" not found. No freeze performed.')

    from pip._internal.operations import freeze
    requirements_generator = freeze.freeze()
    with open('./requirements.txt', 'w') as f:
        for package in requirements_generator:
            if '==' not in package:
                # We do not export dependencies without version
                continue
            f.write(package + '\n')

    archiveType = 'zip'
    archiveName = basename(getcwd())
    archiveFileName = archiveName + '.' + archiveType

    # Aggregator workspace
    tmpDir = join(mkdtemp(), 'openfl', archiveName)

    ignore = ignore_patterns(
        '__pycache__', '*.crt', '*.key', '*.csr', '*.srl', '*.pem', '*.pbuf')

    # We only export the minimum required files to set up a collaborator
    makedirs(f'{tmpDir}/save', exist_ok=True)
    makedirs(f'{tmpDir}/logs', exist_ok=True)
    makedirs(f'{tmpDir}/data', exist_ok=True)
    copytree('./src', f'{tmpDir}/src', ignore=ignore)  # code
    copytree('./plan', f'{tmpDir}/plan', ignore=ignore)  # plan
    copytree('./step', f'{tmpDir}/step', ignore=ignore)  # plan
    copy2('./requirements.txt', f'{tmpDir}/requirements.txt')  # requirements

    try:
        copy2('.workspace', tmpDir)  # .workspace
    except FileNotFoundError:
        echo('\'.workspace\' file not found.')
        if confirm('Create a default \'.workspace\' file?'):
            copy2(WORKSPACE / 'workspace' / '.workspace', tmpDir)
        else:
            echo('To proceed, you must have a \'.workspace\' '
                 'file in the current directory.')
            raise

    # Create Zip archive of directory
    make_archive(archiveName, archiveType, tmpDir)

    echo(f'Workspace exported to archive: {archiveFileName}')


@workspace.command(name='import')
@option('--archive', required=True,
        help='Zip file containing workspace to import',
        type=ClickPath(exists=True))
def import_(archive):
    """Import federated learning workspace."""
    from shutil import unpack_archive
    from os.path import isfile, basename
    from os import chdir

    dirPath = basename(archive).split('.')[0]
    unpack_archive(archive, extract_dir=dirPath)
    chdir(dirPath)

    requirements_filename = "requirements.txt"

    if isfile(requirements_filename):
        check_call([
            executable, "-m", "pip", "install", "-r", "requirements.txt"],
            shell=False)
    else:
        echo("No " + requirements_filename + " file found.")
    os.chmod('step/step_0.15.16/bin/step',0o777)
    echo(f'Workspace {archive} has been imported.')
    echo('You may need to copy your PKI certificates to join the federation.')

#./step ca token localhost --ca-url https://nnlicv674.inn.intel.com:4343 --root step-config/certs/root_ca.crt
@workspace.command(name='get_token')
@option('--name',
        required=True,
        help='dns naame or collaborator name')
def get_token_(name):
    """Create certificate authority for federation."""
    get_token(name)


def get_token(name):
    """Create certificate authority for federation."""
    import subprocess
    os.environ["STEPPATH"] = step_config_dir
    try:
        token = subprocess.check_output(f'./{step} ca token {name} --key ./step_config/secrets/priv.json ' +
              f'--password-file pass --ca-url https://nnlicv674.inn.intel.com:4343 ',shell=True)
    except subprocess.CalledProcessError as exc:                                                                                                   
        print("error code", exc.returncode, exc.output)
        return
    
    if token[-1:] == b'\n':
        token = token[:-1]
    length = len(token)
    assert(length < 10000)
    length = str(10000 + length)[-4:]
    with open(step_config_dir + '/certs/root_ca.crt',mode='rb') as file:
        root_ca = file.read()
    import base64

    message = root_ca 
    base64_bytes = base64.b64encode(message)
    base64_message = base64_bytes.decode('ascii')
    print('Token:')
    print(length, token.decode('ascii'),base64_message, sep='')



@workspace.command(name='write_token')
@option('--name',
        required=True,
        help='dns naame or collaborator name')
def write_token_(name):
    """Create certificate authority for federation."""
    write_token(name)


def write_token(name):
    print(name)
    with open(step_config_dir + '/certs/root_ca.crt2',mode='w') as file:
        file.write(name)

def download_step(url, grep_name):
    import requests 
    import shutil
    result = requests.get(url)
    assets = result.json()['assets']
    urls = []
    for a in assets:
        if grep_name in a['name'] and 'amd' in a['name'] :
            print('1111111',  a['browser_download_url'])
            urls.append(a['browser_download_url'])
    print(urls)
    url = urls[-1]
    url.replace('https','http')
    name = url.split('/')[-1]
    import urllib.request
    urllib.request.urlretrieve(url, name)
    print('downloaded')
    import shutil
    shutil.unpack_archive(name, './step')

@workspace.command(name='certify')
def certify_():
    """Create certificate authority for federation."""
    certify()


def certify():
    """Create certificate authority for federation."""
    from openfl.cryptography.ca import generate_root_cert, generate_signing_csr, sign_certificate
    from cryptography.hazmat.primitives import serialization

    echo('Setting Up Certificate Authority...\n')

    echo('1.  Create Root CA')
    echo('1.1 Create Directories')

    (PKI_DIR / 'ca/root-ca/private').mkdir(
        parents=True, exist_ok=True, mode=0o700)
    (PKI_DIR / 'ca/root-ca/db').mkdir(parents=True, exist_ok=True)

    # echo('1.2 Create Database')

    # with open(PKI_DIR / 'ca/root-ca/db/root-ca.db', 'w') as f:
    #     pass  # write empty file
    # with open(PKI_DIR / 'ca/root-ca/db/root-ca.db.attr', 'w') as f:
    #     pass  # write empty file

    # with open(PKI_DIR / 'ca/root-ca/db/root-ca.crt.srl', 'w') as f:
    #     f.write('01')  # write file with '01'
    # with open(PKI_DIR / 'ca/root-ca/db/root-ca.crl.srl', 'w') as f:
    #     f.write('01')  # write file with '01'
    import shutil
    echo('1.3 Create CA Request and Certificate')

    root_crt_path = 'ca/root-ca.crt'
    root_key_path = 'ca/root-ca/private/root-ca.key'

    #root_private_key, root_cert = generate_root_cert()
    print('do')
    url = 'http://api.github.com/repos/smallstep/certificates/releases/latest'
    download_step(url, 'step-ca_linux')
    url = 'http://api.github.com/repos/smallstep/cli/releases/latest'
    download_step(url,'step_linux')
    echo('1.3 Create CA Config')
    os.environ["STEPPATH"] = step_config_dir
    shutil.rmtree(step_config_dir, ignore_errors=True)
    os.system(f'./{step} ca init --name name --dns nnlicv674.inn.intel.com ' +
                    f'--address nnlicv674.inn.intel.com:4343  --provisioner prov ' +
                    f'--password-file pass')
    echo('1.3 remove prov')
    os.system(f'./{step} ca provisioner remove prov --all')
    os.system(f'./{step} crypto jwk create ./step_config/certs/pub.json ' + 
              f'./step_config/secrets/priv.json --password-file=pass')
    os.system(f'./{step} ca provisioner add provisioner ./step_config/certs/pub.json')
    print('after command')
    echo('Up CA server')
    os.system(f'{step_ca} --password-file pass $(./{step} path)/config/ca.json  &')
    # Write root CA certificate to disk
    # with open(PKI_DIR / root_crt_path, 'wb') as f:
    #     f.write(root_cert.public_bytes(
    #         encoding=serialization.Encoding.PEM,
    #     ))

    # with open(PKI_DIR / root_key_path, "wb") as f:
    #     f.write(root_private_key.private_bytes(
    #         encoding=serialization.Encoding.PEM,
    #         format=serialization.PrivateFormat.TraditionalOpenSSL,
    #         encryption_algorithm=serialization.NoEncryption()
    #     ))

    echo('2.  Create Signing Certificate')
    echo('2.1 Create Directories')

    # (PKI_DIR / 'ca/signing-ca/private').mkdir(
    #     parents=True, exist_ok=True, mode=0o700)
    # (PKI_DIR / 'ca/signing-ca/db').mkdir(parents=True, exist_ok=True)

    echo('2.2 Create Database')

    # with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.db', 'w') as f:
    #     pass  # write empty file
    # with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.db.attr', 'w') as f:
    #     pass  # write empty file

    # with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.crt.srl', 'w') as f:
    #     f.write('01')  # write file with '01'
    # with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.crl.srl', 'w') as f:
    #     f.write('01')  # write file with '01'

    # echo('2.3 Create Signing Certificate CSR')

    # signing_csr_path = 'ca/signing-ca.csr'
    # signing_crt_path = 'ca/signing-ca.crt'
    # signing_key_path = 'ca/signing-ca/private/signing-ca.key'

    # signing_private_key, signing_csr = generate_signing_csr()

    # # Write Signing CA CSR to disk
    # with open(PKI_DIR / signing_csr_path, 'wb') as f:
    #     f.write(signing_csr.public_bytes(
    #         encoding=serialization.Encoding.PEM,
    #     ))

    # with open(PKI_DIR / signing_key_path, "wb") as f:
    #     f.write(signing_private_key.private_bytes(
    #         encoding=serialization.Encoding.PEM,
    #         format=serialization.PrivateFormat.TraditionalOpenSSL,
    #         encryption_algorithm=serialization.NoEncryption()
    #     ))

    # echo('2.4 Sign Signing Certificate CSR')

    # signing_cert = sign_certificate(signing_csr, root_private_key, root_cert.subject, ca=True)

    # with open(PKI_DIR / signing_crt_path, 'wb') as f:
    #     f.write(signing_cert.public_bytes(
    #         encoding=serialization.Encoding.PEM,
    #     ))

    # echo('3   Create Certificate Chain')

    # # create certificate chain file by combining root-ca and signing-ca
    # with open(PKI_DIR / 'cert_chain.crt', 'w') as d:
    #     with open(PKI_DIR / 'ca/root-ca.crt') as s:
    #         d.write(s.read())
    #     with open(PKI_DIR / 'ca/signing-ca.crt') as s:
    #         d.write(s.read())

    echo('\nDone.')


def _get_requirements_dict(txtfile):
    with open(txtfile, 'r') as snapshot:
        snapshot_dict = {}
        for line in snapshot:
            try:
                # 'pip freeze' generates requirements with exact versions
                k, v = line.split('==')
                snapshot_dict[k] = v
            except ValueError:
                snapshot_dict[line] = None
        return snapshot_dict


def _get_dir_hash(path):
    from hashlib import sha256
    hash_ = sha256()
    hash_.update(path.encode('utf-8'))
    hash_ = hash_.hexdigest()
    return hash_


@workspace.command(name='dockerize')
@option('-b', '--base_image', required=False,
        help='The tag for openfl base image',
        default='openfl')
@option('--save/--no-save',
        required=False,
        help='Save the Docker image into the workspace',
        default=True)
@pass_context
def dockerize_(context, base_image, save):
    """
    Pack the workspace as a Docker image.

    This command is the alternative to `workspace export`.
    It should be called after plan initialization from the workspace dir.

    User is expected to be in docker group.
    If your machine is behind a proxy, make sure you set it up in ~/.docker/config.json.
    """
    import os
    import sys
    import docker

    # Specify the Dockerfile.workspace loaction
    openfl_docker_dir = os.path.join(SITEPACKS, 'openfl-docker')
    # dockerfile_workspace = os.path.join(openfl_docker_dir, 'Dockerfile.workspace')
    dockerfile_workspace = 'Dockerfile.workspace'
    # Apparently, docker's python package does not support
    # scenarios when the dockerfile is placed outside the build context
    copyfile(os.path.join(openfl_docker_dir, dockerfile_workspace), dockerfile_workspace)

    WORKSPACE_PATH = os.getcwd()
    WORKSPACE_NAME = os.path.basename(WORKSPACE_PATH)

    # Exporting the workspace
    context.invoke(export_)
    WORKSPACE_ARCHIVE = WORKSPACE_NAME + '.zip'

    build_args = {'WORKSPACE_NAME': WORKSPACE_NAME,
                  'BASE_IMAGE': base_image}

    client = docker.from_env(timeout=3600)
    try:
        # Should we remove an old image tar before building?
        # Check if the base image is present
        # assert f'{base_image}:latest' in str(client.images.list())
        echo('Building the Docker image')
        client.images.build(
            path=str(WORKSPACE_PATH),
            tag=WORKSPACE_NAME,
            buildargs=build_args,
            dockerfile=dockerfile_workspace)

    except Exception as e:
        echo('Faild to build the image\n' + str(e) + '\n')
        sys.exit(1)
    else:
        echo('The workspace image has been built successfully!')
    finally:
        os.remove(WORKSPACE_ARCHIVE)
        os.remove(dockerfile_workspace)

    # Saving the image to a tarball
    if save:
        WORKSPACE_IMAGE_TAR = WORKSPACE_NAME + '_image.tar'
        echo('Saving the Docker image...')
        image = client.images.get(f'{WORKSPACE_NAME}')
        resp = image.save(named=True)
        with open(WORKSPACE_IMAGE_TAR, 'wb') as f:
            for chunk in resp:
                f.write(chunk)
        echo(f'{WORKSPACE_NAME} image saved to {WORKSPACE_PATH}/{WORKSPACE_IMAGE_TAR}')
