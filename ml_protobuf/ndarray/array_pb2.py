# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ndarray.proto

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ndarray.proto',
  package='ml_protobuf.protos.array',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rndarray.proto\x12\x18ml_protobuf.protos.array\"\x80\x01\n\x07NDArray\x12\r\n\x05shape\x18\x01 \x03(\x05\x12.\n\x05\x64type\x18\x02 \x01(\x0e\x32\x1f.ml_protobuf.protos.array.DType\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\x13\n\x0bstring_data\x18\x04 \x03(\t\x12\x13\n\x0bnone_object\x18\x05 \x03(\x08*\xa0\x01\n\x05\x44Type\x12\x0b\n\x07\x46LOAT16\x10\x00\x12\x0b\n\x07\x46LOAT32\x10\x01\x12\x0b\n\x07\x46LOAT64\x10\x02\x12\t\n\x05INT16\x10\x03\x12\t\n\x05INT32\x10\x04\x12\t\n\x05INT64\x10\x05\x12\t\n\x05UINT8\x10\x06\x12\n\n\x06UINT16\x10\x07\x12\n\n\x06UINT32\x10\x08\x12\n\n\x06UINT64\x10\t\x12\x08\n\x04\x42OOL\x10\n\x12\n\n\x06STRING\x10\x0b\x12\n\n\x06OBJECT\x10\x0c\x62\x06proto3'
)

_DTYPE = _descriptor.EnumDescriptor(
  name='DType',
  full_name='ml_protobuf.protos.array.DType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FLOAT16', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FLOAT32', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FLOAT64', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INT16', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INT32', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INT64', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UINT8', index=6, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UINT16', index=7, number=7,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UINT32', index=8, number=8,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='UINT64', index=9, number=9,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BOOL', index=10, number=10,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='STRING', index=11, number=11,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='OBJECT', index=12, number=12,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=175,
  serialized_end=335,
)
_sym_db.RegisterEnumDescriptor(_DTYPE)

DType = enum_type_wrapper.EnumTypeWrapper(_DTYPE)
FLOAT16 = 0
FLOAT32 = 1
FLOAT64 = 2
INT16 = 3
INT32 = 4
INT64 = 5
UINT8 = 6
UINT16 = 7
UINT32 = 8
UINT64 = 9
BOOL = 10
STRING = 11
OBJECT = 12



_NDARRAY = _descriptor.Descriptor(
  name='NDArray',
  full_name='ml_protobuf.protos.array.NDArray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='ml_protobuf.protos.array.NDArray.shape', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='ml_protobuf.protos.array.NDArray.dtype', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data', full_name='ml_protobuf.protos.array.NDArray.data', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='string_data', full_name='ml_protobuf.protos.array.NDArray.string_data', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='none_object', full_name='ml_protobuf.protos.array.NDArray.none_object', index=4,
      number=5, type=8, cpp_type=7, label=3,
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
  serialized_start=44,
  serialized_end=172,
)

_NDARRAY.fields_by_name['dtype'].enum_type = _DTYPE
DESCRIPTOR.message_types_by_name['NDArray'] = _NDARRAY
DESCRIPTOR.enum_types_by_name['DType'] = _DTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NDArray = _reflection.GeneratedProtocolMessageType('NDArray', (_message.Message,), {
  'DESCRIPTOR' : _NDARRAY,
  '__module__' : 'ndarray_pb2'
  # @@protoc_insertion_point(class_scope:ml_protobuf.protos.array.NDArray)
  })
_sym_db.RegisterMessage(NDArray)


# @@protoc_insertion_point(module_scope)
