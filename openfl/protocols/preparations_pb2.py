# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: preparations.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='preparations.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x12preparations.proto\"U\n\x08NodeInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06\x61\x64ress\x18\x02 \x01(\t\x12\x16\n\x0e\x63uda_available\x18\x03 \x01(\x08\x12\x13\n\x0bmemory_size\x18\x04 \x01(\r\"\x83\x01\n\tShardInfo\x12\x1c\n\tnode_info\x18\x01 \x01(\x0b\x32\t.NodeInfo\x12\x19\n\x11shard_description\x18\x02 \x01(\t\x12\x11\n\tn_samples\x18\x03 \x01(\x04\x12\x14\n\x0csample_shape\x18\x04 \x03(\t\x12\x14\n\x0ctarget_shape\x18\x05 \x03(\t\"(\n\x14ShardAcknowledgement\x12\x10\n\x08\x61\x63\x63\x65pted\x18\x01 \x01(\x08\"\x1c\n\x08Response\x12\x10\n\x08\x61\x63\x63\x65pted\x18\x01 \x01(\x08\"/\n\x0e\x45xperimentData\x12\x0c\n\x04size\x18\x01 \x01(\r\x12\x0f\n\x07npbytes\x18\x02 \x01(\x0c\"d\n\x0e\x45xperimentInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x1a\n\x12\x63ollaborator_names\x18\x02 \x03(\t\x12(\n\x0f\x65xperiment_data\x18\x03 \x01(\x0b\x32\x0f.ExperimentData\"\x1b\n\x19GetRegisterdShardsRequest\"<\n\x1aGetRegisterdShardsResponse\x12\x1e\n\nshard_info\x18\x01 \x03(\x0b\x32\n.ShardInfo\"2\n\x15WaitExperimentRequest\x12\x19\n\x11\x63ollaborator_name\x18\x01 \x01(\t\"1\n\x16WaitExperimentResponse\x12\x17\n\x0f\x65xperiment_name\x18\x01 \x01(\t\"N\n\x18GetExperimentDataRequest\x12\x17\n\x0f\x65xperiment_name\x18\x01 \x01(\t\x12\x19\n\x11\x63ollaborator_name\x18\x02 \x01(\t\"\x16\n\x14GetShardsInfoRequest2\x96\x03\n\x12\x46\x65\x64\x65rationDirector\x12\x37\n\x10\x41\x63knowledgeShard\x12\n.ShardInfo\x1a\x15.ShardAcknowledgement\"\x00\x12G\n\x0eWaitExperiment\x12\x16.WaitExperimentRequest\x1a\x17.WaitExperimentResponse\"\x00(\x01\x30\x01\x12\x43\n\x11GetExperimentData\x12\x19.GetExperimentDataRequest\x1a\x0f.ExperimentData\"\x00\x30\x01\x12\x32\n\x10SetNewExperiment\x12\x0f.ExperimentInfo\x1a\t.Response\"\x00(\x01\x12O\n\x12GetRegisterdShards\x12\x1a.GetRegisterdShardsRequest\x1a\x1b.GetRegisterdShardsResponse\"\x00\x12\x34\n\rGetShardsInfo\x12\x15.GetShardsInfoRequest\x1a\n.ShardInfo\"\x00\x62\x06proto3'
)




_NODEINFO = _descriptor.Descriptor(
  name='NodeInfo',
  full_name='NodeInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='NodeInfo.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='adress', full_name='NodeInfo.adress', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cuda_available', full_name='NodeInfo.cuda_available', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='memory_size', full_name='NodeInfo.memory_size', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=107,
)


_SHARDINFO = _descriptor.Descriptor(
  name='ShardInfo',
  full_name='ShardInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='node_info', full_name='ShardInfo.node_info', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='shard_description', full_name='ShardInfo.shard_description', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='n_samples', full_name='ShardInfo.n_samples', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sample_shape', full_name='ShardInfo.sample_shape', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='target_shape', full_name='ShardInfo.target_shape', index=4,
      number=5, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=110,
  serialized_end=241,
)


_SHARDACKNOWLEDGEMENT = _descriptor.Descriptor(
  name='ShardAcknowledgement',
  full_name='ShardAcknowledgement',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='accepted', full_name='ShardAcknowledgement.accepted', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=243,
  serialized_end=283,
)


_RESPONSE = _descriptor.Descriptor(
  name='Response',
  full_name='Response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='accepted', full_name='Response.accepted', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=285,
  serialized_end=313,
)


_EXPERIMENTDATA = _descriptor.Descriptor(
  name='ExperimentData',
  full_name='ExperimentData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='size', full_name='ExperimentData.size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='npbytes', full_name='ExperimentData.npbytes', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=315,
  serialized_end=362,
)


_EXPERIMENTINFO = _descriptor.Descriptor(
  name='ExperimentInfo',
  full_name='ExperimentInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='ExperimentInfo.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='collaborator_names', full_name='ExperimentInfo.collaborator_names', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='experiment_data', full_name='ExperimentInfo.experiment_data', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=364,
  serialized_end=464,
)


_GETREGISTERDSHARDSREQUEST = _descriptor.Descriptor(
  name='GetRegisterdShardsRequest',
  full_name='GetRegisterdShardsRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=466,
  serialized_end=493,
)


_GETREGISTERDSHARDSRESPONSE = _descriptor.Descriptor(
  name='GetRegisterdShardsResponse',
  full_name='GetRegisterdShardsResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='shard_info', full_name='GetRegisterdShardsResponse.shard_info', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=495,
  serialized_end=555,
)


_WAITEXPERIMENTREQUEST = _descriptor.Descriptor(
  name='WaitExperimentRequest',
  full_name='WaitExperimentRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='collaborator_name', full_name='WaitExperimentRequest.collaborator_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=557,
  serialized_end=607,
)


_WAITEXPERIMENTRESPONSE = _descriptor.Descriptor(
  name='WaitExperimentResponse',
  full_name='WaitExperimentResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='experiment_name', full_name='WaitExperimentResponse.experiment_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=609,
  serialized_end=658,
)


_GETEXPERIMENTDATAREQUEST = _descriptor.Descriptor(
  name='GetExperimentDataRequest',
  full_name='GetExperimentDataRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='experiment_name', full_name='GetExperimentDataRequest.experiment_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='collaborator_name', full_name='GetExperimentDataRequest.collaborator_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=660,
  serialized_end=738,
)


_GETSHARDSINFOREQUEST = _descriptor.Descriptor(
  name='GetShardsInfoRequest',
  full_name='GetShardsInfoRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=740,
  serialized_end=762,
)

_SHARDINFO.fields_by_name['node_info'].message_type = _NODEINFO
_EXPERIMENTINFO.fields_by_name['experiment_data'].message_type = _EXPERIMENTDATA
_GETREGISTERDSHARDSRESPONSE.fields_by_name['shard_info'].message_type = _SHARDINFO
DESCRIPTOR.message_types_by_name['NodeInfo'] = _NODEINFO
DESCRIPTOR.message_types_by_name['ShardInfo'] = _SHARDINFO
DESCRIPTOR.message_types_by_name['ShardAcknowledgement'] = _SHARDACKNOWLEDGEMENT
DESCRIPTOR.message_types_by_name['Response'] = _RESPONSE
DESCRIPTOR.message_types_by_name['ExperimentData'] = _EXPERIMENTDATA
DESCRIPTOR.message_types_by_name['ExperimentInfo'] = _EXPERIMENTINFO
DESCRIPTOR.message_types_by_name['GetRegisterdShardsRequest'] = _GETREGISTERDSHARDSREQUEST
DESCRIPTOR.message_types_by_name['GetRegisterdShardsResponse'] = _GETREGISTERDSHARDSRESPONSE
DESCRIPTOR.message_types_by_name['WaitExperimentRequest'] = _WAITEXPERIMENTREQUEST
DESCRIPTOR.message_types_by_name['WaitExperimentResponse'] = _WAITEXPERIMENTRESPONSE
DESCRIPTOR.message_types_by_name['GetExperimentDataRequest'] = _GETEXPERIMENTDATAREQUEST
DESCRIPTOR.message_types_by_name['GetShardsInfoRequest'] = _GETSHARDSINFOREQUEST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NodeInfo = _reflection.GeneratedProtocolMessageType('NodeInfo', (_message.Message,), {
  'DESCRIPTOR' : _NODEINFO,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:NodeInfo)
  })
_sym_db.RegisterMessage(NodeInfo)

ShardInfo = _reflection.GeneratedProtocolMessageType('ShardInfo', (_message.Message,), {
  'DESCRIPTOR' : _SHARDINFO,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:ShardInfo)
  })
_sym_db.RegisterMessage(ShardInfo)

ShardAcknowledgement = _reflection.GeneratedProtocolMessageType('ShardAcknowledgement', (_message.Message,), {
  'DESCRIPTOR' : _SHARDACKNOWLEDGEMENT,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:ShardAcknowledgement)
  })
_sym_db.RegisterMessage(ShardAcknowledgement)

Response = _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSE,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:Response)
  })
_sym_db.RegisterMessage(Response)

ExperimentData = _reflection.GeneratedProtocolMessageType('ExperimentData', (_message.Message,), {
  'DESCRIPTOR' : _EXPERIMENTDATA,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:ExperimentData)
  })
_sym_db.RegisterMessage(ExperimentData)

ExperimentInfo = _reflection.GeneratedProtocolMessageType('ExperimentInfo', (_message.Message,), {
  'DESCRIPTOR' : _EXPERIMENTINFO,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:ExperimentInfo)
  })
_sym_db.RegisterMessage(ExperimentInfo)

GetRegisterdShardsRequest = _reflection.GeneratedProtocolMessageType('GetRegisterdShardsRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETREGISTERDSHARDSREQUEST,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:GetRegisterdShardsRequest)
  })
_sym_db.RegisterMessage(GetRegisterdShardsRequest)

GetRegisterdShardsResponse = _reflection.GeneratedProtocolMessageType('GetRegisterdShardsResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETREGISTERDSHARDSRESPONSE,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:GetRegisterdShardsResponse)
  })
_sym_db.RegisterMessage(GetRegisterdShardsResponse)

WaitExperimentRequest = _reflection.GeneratedProtocolMessageType('WaitExperimentRequest', (_message.Message,), {
  'DESCRIPTOR' : _WAITEXPERIMENTREQUEST,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:WaitExperimentRequest)
  })
_sym_db.RegisterMessage(WaitExperimentRequest)

WaitExperimentResponse = _reflection.GeneratedProtocolMessageType('WaitExperimentResponse', (_message.Message,), {
  'DESCRIPTOR' : _WAITEXPERIMENTRESPONSE,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:WaitExperimentResponse)
  })
_sym_db.RegisterMessage(WaitExperimentResponse)

GetExperimentDataRequest = _reflection.GeneratedProtocolMessageType('GetExperimentDataRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETEXPERIMENTDATAREQUEST,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:GetExperimentDataRequest)
  })
_sym_db.RegisterMessage(GetExperimentDataRequest)

GetShardsInfoRequest = _reflection.GeneratedProtocolMessageType('GetShardsInfoRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETSHARDSINFOREQUEST,
  '__module__' : 'preparations_pb2'
  # @@protoc_insertion_point(class_scope:GetShardsInfoRequest)
  })
_sym_db.RegisterMessage(GetShardsInfoRequest)



_FEDERATIONDIRECTOR = _descriptor.ServiceDescriptor(
  name='FederationDirector',
  full_name='FederationDirector',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=765,
  serialized_end=1171,
  methods=[
  _descriptor.MethodDescriptor(
    name='AcknowledgeShard',
    full_name='FederationDirector.AcknowledgeShard',
    index=0,
    containing_service=None,
    input_type=_SHARDINFO,
    output_type=_SHARDACKNOWLEDGEMENT,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='WaitExperiment',
    full_name='FederationDirector.WaitExperiment',
    index=1,
    containing_service=None,
    input_type=_WAITEXPERIMENTREQUEST,
    output_type=_WAITEXPERIMENTRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetExperimentData',
    full_name='FederationDirector.GetExperimentData',
    index=2,
    containing_service=None,
    input_type=_GETEXPERIMENTDATAREQUEST,
    output_type=_EXPERIMENTDATA,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetNewExperiment',
    full_name='FederationDirector.SetNewExperiment',
    index=3,
    containing_service=None,
    input_type=_EXPERIMENTINFO,
    output_type=_RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetRegisterdShards',
    full_name='FederationDirector.GetRegisterdShards',
    index=4,
    containing_service=None,
    input_type=_GETREGISTERDSHARDSREQUEST,
    output_type=_GETREGISTERDSHARDSRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetShardsInfo',
    full_name='FederationDirector.GetShardsInfo',
    index=5,
    containing_service=None,
    input_type=_GETSHARDSINFOREQUEST,
    output_type=_SHARDINFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_FEDERATIONDIRECTOR)

DESCRIPTOR.services_by_name['FederationDirector'] = _FEDERATIONDIRECTOR

# @@protoc_insertion_point(module_scope)
