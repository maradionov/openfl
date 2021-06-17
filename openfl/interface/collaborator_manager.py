import logging
from importlib import import_module
from os import path
import sys
from pathlib import Path
import shutil

import click
from click import group, option, pass_context
from click import Path as ClickPath
from yaml import safe_load

from openfl.interface.cli_helper import WORKSPACE
from openfl.interface.cli_helper import SITEPACKS
from openfl.component.collaborator_manager.collaborator_manager import CollaboratorManager

logger = logging.getLogger(__name__)


@group()
@pass_context
def collaborator_manager(context):
    """Manage Federated Learning Envoy."""
    context.obj['group'] = 'collaborator-manager'


@collaborator_manager.command(name='start')
@option('-n', '--shard-name', required=True,
        help='Current shard name')
@option('-d', '--director-uri', required=True,
        help='The FQDN of the federation director')
@option('-sc', '--shard-config-path', default='shard_config.yaml',
        help='The shard config path', type=ClickPath(exists=True))
def start_(shard_name, director_uri, shard_config_path):
    """Start the collaborator manager."""
    logger.info('🧿 Starting the Collaborator Manager.')

    shard_descriptor = shard_descriptor_from_config(shard_config_path)
    keeper = CollaboratorManager(shard_name=shard_name, director_uri=director_uri,
                                 shard_descriptor=shard_descriptor)

    keeper.start()


@collaborator_manager.command(name='create-workspace')
@option('-p', '--collaborator-manager-path', required=True,
        help='The Collaborator Manager path', type=ClickPath())
def create(collaborator_manager_path):
    """Create a collaborator manager workspace."""
    collaborator_manager_path = Path(collaborator_manager_path)
    if collaborator_manager_path.exists():
        if not click.confirm('Collaborator manager workspace already exists. Recreate?',
                             default=True):
            sys.exit(1)
        shutil.rmtree(collaborator_manager_path)
    (collaborator_manager_path / 'cert').mkdir(parents=True, exist_ok=True)
    (collaborator_manager_path / 'logs').mkdir(parents=True, exist_ok=True)
    (collaborator_manager_path / 'data').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WORKSPACE / 'default/shard_config.yaml',
                    collaborator_manager_path / 'shard_config.yaml')
    shutil.copyfile(SITEPACKS / 'openfl/component/collaborator_manager/shard_descriptor.py',
                    collaborator_manager_path / 'shard_descriptor.py')


def shard_descriptor_from_config(shard_config_path: str):
    with open(shard_config_path) as stream:
        shard_config = safe_load(stream)
    class_name = path.splitext(shard_config['template'])[1].strip('.')
    module_path = path.splitext(shard_config['template'])[0]
    params = shard_config['params']

    module = import_module(module_path)
    instance = getattr(module, class_name)(**params)

    return instance
