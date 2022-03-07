#!/usr/bin/env python
"""Protoc Plugin to generate protoplus files. Loosely based on mypy's mypy-protobuf implementation"""
from __future__ import annotations

import collections
import contextlib
import functools
import os.path as op
import pathlib
import re
import warnings
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

import itertools
import sys
from itertools import groupby, chain

try:
    import importlib.metadata as importlib_metadata
except ImportError:  # for Python<3.8
    import importlib_metadata

import google.protobuf.descriptor_pb2 as d
from google.protobuf.compiler import plugin_pb2 as plugin_pb2
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from google.protobuf.internal.well_known_types import WKTBASES
from google.protobuf.message import DecodeError

__dist__ = 'codestare-proto-plus'
try:
    __version__ = importlib_metadata.version(__dist__)
except importlib_metadata.PackageNotFoundError:
    print(f"Distribution {__dist__} is not installed.")

# SourceCodeLocation is defined by `message Location` here
# https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/descriptor.proto
SourceCodeLocation = List[int]

# So phabricator doesn't think codestare-proto-plus.py is generated
GENERATED = "@ge" + "nerated"
CONTENT = f"{GENERATED} by codestare-proto-plus.  Do not edit manually!"

HEADER_TEMPLATE = """\"\"\"
{}
\"\"\"
"""
HEADER = HEADER_TEMPLATE.format(CONTENT)

INIT_HEADER = HEADER_TEMPLATE.format(
    CONTENT + ("\n\n"
               "Note:\n"
               "    __init__.py files for packages are autogenerated, and "
               "import all messages from submodules to keep import "
               "names consistent with the directory structure of the ``.proto`` source files."
               )
)

# See https://github.com/dropbox/mypy-protobuf/issues/73 for details
PYTHON_RESERVED = {
    "False",
    "None",
    "True",
    "and",
    "as",
    "async",
    "await",
    "assert",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}

PROTO_ENUM_RESERVED = {
    "Name",
    "Value",
    "keys",
    "values",
    "items",
}

PARAMETER_FLAGS = [
    'readable_imports',
    'relative_imports',
    'generate_inits',
    'quiet',
]

PARAMETER_KEYWORD_ARGS = [
    'save_request',
    'package',
]

MAX_LINE_LEN = 80


# def is_scalar(fd: d.FieldDescriptorProto) -> bool:
#     return not (
#             fd.type == d.FieldDescriptorProto.TYPE_MESSAGE
#             or fd.type == d.FieldDescriptorProto.TYPE_GROUP
#     )


def removeprefix(value: str, prefix=''):
    if sys.version_info >= (3, 9):
        return value.removeprefix(prefix)
    else:
        return value[len(prefix):] if value.startswith(prefix) else value


class Descriptors(object):
    def __init__(self, request: plugin_pb2.CodeGeneratorRequest) -> None:
        files = {f.name: f for f in request.proto_file}
        to_generate = {n: files[n] for n in request.file_to_generate}
        self.files: Dict[str, d.FileDescriptorProto] = files
        self.to_generate: Dict[str, d.FileDescriptorProto] = to_generate
        self.messages: Dict[str, d.DescriptorProto] = {}
        self.message_to_fd: Dict[str, d.FileDescriptorProto] = {}

        def _add_enums(
                enums: "RepeatedCompositeFieldContainer[d.EnumDescriptorProto]",
                prefix: str,
                _fd: d.FileDescriptorProto,
        ) -> None:
            for enum in enums:
                self.message_to_fd[prefix + enum.name] = _fd
                self.message_to_fd[prefix + enum.name + ".ValueType"] = _fd

        def _add_messages(
                messages: "RepeatedCompositeFieldContainer[d.DescriptorProto]",
                prefix: str,
                _fd: d.FileDescriptorProto,
        ) -> None:
            for message in messages:
                self.messages[prefix + message.name] = message
                self.message_to_fd[prefix + message.name] = _fd
                sub_prefix = prefix + message.name + "."
                _add_messages(message.nested_type, sub_prefix, _fd)
                _add_enums(message.enum_type, sub_prefix, _fd)

        for fd in request.proto_file:
            start_prefix = "." + fd.package + "." if fd.package else "."
            _add_messages(fd.message_type, start_prefix, fd)
            _add_enums(fd.enum_type, start_prefix, fd)


class Writer(object):
    def __init__(
            self,
            *,
            fd: d.FileDescriptorProto,
            descriptors: Descriptors,
            readable_imports: bool = False,
            relative_imports: bool = False,
            package: str | None = None,
    ) -> None:
        self.fd = fd
        self.descriptors = descriptors
        self.package = package
        self.readable_imports = readable_imports
        self.relative_imports = relative_imports

        self.lines: List[str] = []
        self.indent = ""

        # Set of {x}, where {x} corresponds to to `import {x}`
        self.imports: Set[Tuple[str, Optional[str]]] = set()
        # dictionary of x->(y,z) for `from {x} import {y} as {z}`
        # if {z} is None, then it shortens to `from {x} import {y}`
        self.from_imports: Dict[str, Set[Tuple[str, Optional[str]]]] = collections.defaultdict(set)

    def _import(self, path: str, name: str, alias=None) -> str:
        """Imports a path and returns a handle to it
        eg. self._import("typing", "Optional") -> "Optional"
        """
        imp = path.replace("/", ".")

        if self.readable_imports:
            self.from_imports[imp].add((name, alias))
            return alias or name
        else:
            self.imports.add((imp, alias))
            return (alias or imp) + "." + name

    def _builtin(self, name: str) -> str:
        return self._import("builtins", name)

    @contextlib.contextmanager
    def _indent(self) -> Iterator[None]:
        self.indent = self.indent + "    "
        yield
        self.indent = self.indent[:-4]

    def _write_line(self, line: str, *args: Any, break_line=False, dry_run=False) -> List[str]:
        if args:
            line = line.format(*args)

        lines = [] if break_line else [line]
        if not lines:
            indent = '    '
            remainder = self.indent + line

            while len(remainder) > MAX_LINE_LEN:
                remainder = remainder[len(self.indent):]

                break_at = remainder.rfind(' ', 0, MAX_LINE_LEN)
                if break_at <= len(indent):
                    remainder = self.indent + remainder
                    break

                lines.append(remainder[:break_at])
                remaining = remainder[break_at:]
                remainder = self.indent + indent + (remaining[1:] if remaining.startswith(' ') else remaining)

            lines.append(remainder[len(self.indent):])
            # if lines:
            #     lines = lines[:1] + ['    ' + l for l in lines[1:]]

        storage = [] if dry_run else self.lines

        for line in lines:
            if line == "":
                storage.append(line)
            else:
                storage.append(self.indent + line)

        return storage

    def _break_text(self, text_block: str) -> List[str]:
        if text_block == "":
            return []
        return [
            l[1:] if l.startswith(" ") else l for l in text_block.rstrip().split("\n")
        ]

    @abstractmethod
    def write_module_attributes(self) -> None:
        ...

    def write(self) -> str:
        import_lines = []

        for pkg, alias in sorted(self.imports):
            import_lines.append(f"import {pkg}" if alias is None else f"import {pkg} as {alias}")

        for pkg, items in sorted(self.from_imports.items()):
            import_lines.append(f"from {pkg} import (")
            for (name, reexport_name) in sorted(items):
                if reexport_name is None:
                    import_lines.append(f"    {name},")
                else:
                    import_lines.append(f"    {name} as {reexport_name},")
            import_lines.append(")\n")
        import_lines.append("")

        return "\n".join(import_lines + self.lines)

    def _import_message(self, name: str, use_alias=False) -> str:
        """Import a referenced message and return a handle"""
        message_fd = self.descriptors.message_to_fd[name]
        assert message_fd.name.endswith(".proto")

        # Strip off package name
        if message_fd.package:
            assert name.startswith("." + message_fd.package + ".")
            name = name[len("." + message_fd.package + "."):]
        else:
            assert name.startswith(".")
            name = name[1:]

        # Use prepended "_r_" to disambiguate message names that alias python reserved keywords
        split = name.split(".")
        for i, part in enumerate(split):
            if part in PYTHON_RESERVED:
                split[i] = "_r_" + part
        name = ".".join(split)

        # Message defined in this file.
        if message_fd.name == self.fd.name:
            return name

        # Not in file. Must import
        # Python generated code ignores proto packages, so the only relevant factor is
        # whether it is in the file or not.
        package_name = message_fd.name[:-6].replace("-", "_") + "_pb_plus"
        pkgs = package_name.split('/')
        prefix = ''.join(pkg[0] for pkg in pkgs[:-1])
        alias = None if self.readable_imports else f"{prefix}_{pkgs[-1]}"

        import_name = self._import(
            package_name, split[0],
            alias=alias if use_alias else None
        )

        remains = ".".join(split[1:])
        if not remains:
            return import_name

        # remains could either be a direct import of a nested enum or message
        # from another package.
        return import_name + "." + remains


class PkgWriter(Writer):
    """Writes a single pb_plus.py file"""

    _current_working_class = None
    field_matcher = re.compile(r'(\w+) = ([^\s]*Field)\((.+?)\)', re.MULTILINE | re.DOTALL)
    info_matcher = re.compile(r'\s*([^\s]+),\s+.*', re.DOTALL)

    @contextlib.contextmanager
    def working_class(self, value):
        previous = self._current_working_class
        self._current_working_class = value
        yield self._current_working_class
        self._current_working_class = previous

    def __init__(self, *, fd: d.FileDescriptorProto, **kwargs) -> None:
        super().__init__(fd=fd, **kwargs)
        self.source_code_info_by_scl = {
            tuple(location.path): location for location in fd.source_code_info.location
        }

    def _fix_meta_import(self, import_name, mapping: Dict[str, str]):
        """
        Some packages are not imported from the module where they are actually defined.
        (e.g. 'proto.Field')
        """
        for meta_module, defining_module in mapping.items():
            if import_name.startswith(meta_module) and not import_name.startswith(defining_module):
                import_name = defining_module + import_name[len(meta_module):]
                break

        return import_name

    def _get_doc_import_name(self, full_spec):
        specs = full_spec.rsplit('.', maxsplit=1)
        module, name = (None, specs[0]) if len(specs) == 1 else specs

        imports = [
            f"{imp}.{name}" for imp, alias in self.imports
            if (alias and alias == name) or imp == module
        ]

        imports = imports or [
            f"{imp}.{name}" for imp, imports in self.from_imports.items()
            if any((alias and alias == name) or imp_name == name for imp_name, alias in imports)
        ]

        if not imports:
            # Name is defined in same module
            return f".{name}"

        assert len(imports) == 1, f"{len(imports)} imports define {full_spec}"

        return imports[0]

    def _generate_docstring_comment_(self, header: List, field_infos: Dict, label_prefix):
        def _get_insert_pos(value):
            return [index + 1 for index, line in enumerate(comment) if re.match(r'^\s*{}$'.format(value), line)]

        info_comment_template = "{name} ({field_class}): :obj:`~{field_class}` of type :obj:`~{field_type}`"
        wl = functools.partial(self._write_line, dry_run=True, break_line=True)
        label_prefix = label_prefix[1:].lower()

        top_level = header
        oneof_header = ".. admonition:: One Ofs"
        attributes_header = 'Attributes:'

        if any('oneof' in argmap for argmap in field_infos.values()):
            for has_oneof in [argmap for argmap in field_infos.values() if 'oneof' in argmap]:
                has_oneof['oneof'] = has_oneof['oneof'].strip("'")

            header += [oneof_header] if not header[-1] else ['', oneof_header]

        if field_infos:
            header += [attributes_header] if not header[-1] else ['', attributes_header]

        comment = self._write_comments(top_level, dry_run=True, break_line=True)

        with self._indent():
            insert_attr = _get_insert_pos(attributes_header)
            if insert_attr:
                body = []
                for name, argmap in field_infos.items():
                    template = info_comment_template
                    if 'oneof' in argmap:
                        template += " -- *oneof* :attr:`.{oneof}`"
                    if 'optional' in argmap:
                        template += ' -- *optional*'

                    lines = [
                                template.format(name=name, label_prefix=label_prefix, **argmap),
                            ] + argmap.get('comment') or []

                    formatted = [
                        l_ for line in lines
                        for l_ in wl(line)
                    ]
                    body.extend(formatted)

                for pos in insert_attr:
                    comment = comment[:pos] + body + comment[pos:]

            insert_oneof = _get_insert_pos(oneof_header)
            if insert_oneof:
                oneofs = set(filter(None, (argmap.get('oneof') for argmap in field_infos.values())))
                oneof_mapping = {
                    oneof: [name for name, argmap in field_infos.items() if argmap.get('oneof') == oneof]
                    for oneof in oneofs
                }
                body = [''] + wl(f"This message defines the following *oneof* group[s]") + ['']

                for name, fields in oneof_mapping.items():
                    body += wl(f".. attribute:: {name}")
                    with self._indent():
                        body += ['']
                        body.extend(
                            itertools.chain.from_iterable(
                                wl(line) for line in map('- \t:attr:`.{}`'.format, fields)
                            )
                        )

                for pos in insert_oneof:
                    comment = comment[:pos] + body + comment[pos:]

        return comment

    def _has_comments(self, scl: SourceCodeLocation) -> bool:
        sci_loc = self.source_code_info_by_scl.get(tuple(scl))
        return sci_loc is not None and bool(
            sci_loc.leading_detached_comments
            or sci_loc.leading_comments
            or sci_loc.trailing_comments
        )

    def _get_comments(self, scl: SourceCodeLocation) -> List[str]:
        if not self._has_comments(scl):
            return []

        sci_loc = self.source_code_info_by_scl.get(tuple(scl))
        assert sci_loc is not None

        lines = []
        for leading_detached_comment in sci_loc.leading_detached_comments:
            lines.extend(self._break_text(leading_detached_comment))
            lines.append("")
        if sci_loc.leading_comments is not None:
            lines.extend(self._break_text(sci_loc.leading_comments))
        # Trailing comments also go in the header - to make sure it gets into the docstring
        if sci_loc.trailing_comments is not None:
            lines.extend(self._break_text(sci_loc.trailing_comments))

        return [
            # Escape triple-quotes that would otherwise end the docstring early.
            line.replace("\\", "\\\\").replace('"""', r"\"\"\"")
            for line in lines
        ]

    def _write_comments(self, lines, **kwargs) -> List[str]:
        written = []
        if not lines:
            return written

        write = functools.partial(self._write_line, **kwargs)
        if len(lines) == 1:
            line = lines[0]
            if line.endswith(('"', "\\")):
                # Docstrings are terminated with triple-quotes, so if the documentation itself ends in a quote,
                # insert some whitespace to separate it from the closing quotes.
                # This is not necessary with multiline comments
                # because in that case we always insert a newline before the trailing triple-quotes.
                line = line + " "
            written += write(f'"""{line}"""')
        else:
            for i, line in enumerate(lines):
                if i == 0:
                    written += write(f'"""{line}')
                else:
                    written += write(line)
            written += write('"""')

        written += write('')
        return written

    def write_enum_values(
            self,
            values: Iterable[Tuple[int, d.EnumValueDescriptorProto]],
            scl_prefix: SourceCodeLocation,
    ) -> None:
        for i, val in values:
            if val.name in PYTHON_RESERVED:
                continue

            scl = scl_prefix + [i]
            self._write_line(
                f"{val.name} = {val.number}",
            )
            if self._write_comments(self._get_comments(scl)):
                self._write_line("")  # Extra newline to separate

    def write_module_attributes(self) -> None:
        l = self._write_line
        module = self._import('proto', 'module')

        l(f"__protobuf__ = {module}(")
        with self._indent():
            l(f'package="{self.package}",')
            l("manifest={")
            with self._indent():
                for message in self.fd.message_type:
                    l(f'"{message.name}",')
                for enum in self.fd.enum_type:
                    l(f'"{enum.name}",')
            l("}")
        l(")\n\n")

    def write_enums(
            self,
            enums: Iterable[d.EnumDescriptorProto],
            scl_prefix: SourceCodeLocation,
    ) -> None:
        l = self._write_line
        for i, enum in enumerate(enums):
            class_name = (
                enum.name if enum.name not in PYTHON_RESERVED else "_r_" + enum.name
            )
            enum_helper_class = self._import('proto', 'Enum')

            l(f"class {class_name}({enum_helper_class}):")
            with self._indent():
                scl = scl_prefix + [i]
                self._write_comments(self._get_comments(scl))
                self.write_enum_values(
                    enumerate(enum.value),
                    scl + [d.EnumDescriptorProto.VALUE_FIELD_NUMBER],
                )
            l("")

    def write_messages(
            self,
            messages: Iterable[d.DescriptorProto],
            scl_prefix: SourceCodeLocation,
    ) -> None:
        l = self._write_line

        for i, desc in enumerate(messages):
            # Reproduce some hardcoded logic from the protobuf implementation - where
            # some specific "well_known_types" generated protos to have additional
            # base classes
            addl_base = ""
            if self.fd.package + "." + desc.name in WKTBASES:
                # chop off the .proto - and import the well known type
                # eg `from google.protobuf.duration import Duration`
                well_known_type = WKTBASES[self.fd.package + "." + desc.name]
                addl_base = ", " + self._import(
                    "google.protobuf.internal.well_known_types",
                    well_known_type.__name__,
                )

            class_name = (
                desc.name if desc.name not in PYTHON_RESERVED else "_r_" + desc.name
            )
            message_class = self._import("proto.message", "Message")

            l(f"class {class_name}({message_class}{addl_base}):")
            _line_count = len(self.lines)

            with self._indent():
                scl = scl_prefix + [i]

                class_comment = self._get_comments(scl)

                # Nested enums/messages
                with self.working_class(desc):
                    self.write_enums(
                        desc.enum_type,
                        scl + [d.DescriptorProto.ENUM_TYPE_FIELD_NUMBER],
                    )
                    self.write_messages(
                        desc.nested_type,
                        scl + [d.DescriptorProto.NESTED_TYPE_FIELD_NUMBER],
                    )

                field_infos = {}

                for idx, field in enumerate(desc.field):
                    if field.name in PYTHON_RESERVED:
                        continue
                    if field.number is None:
                        warnings.warn(f"Field {field} has no number set in .proto file. "
                                      f"It's the {idx + 1} field for {class_name}")

                    field_class, field_type = self.protoplus_type(field)
                    args = {
                        'number': field.number,
                    }
                    if field.HasField('oneof_index'):
                        args['oneof'] = f"'{desc.oneof_decl[field.oneof_index].name}'"
                    else:
                        args['optional']: field.label == field.LABEL_OPTIONAL

                    # write field specification
                    l(f"{field.name} = {field_class}(")
                    if self._current_working_class:
                        field_type = removeprefix(field_type, f"{self._current_working_class.name}.")

                    field_type = removeprefix(field_type, f"{class_name}.")
                    with self._indent():
                        l(f'{field_type},')
                        for k, v in args.items():
                            l(f"{k}={v},")
                    l(")")

                    field_scl = scl + [d.DescriptorProto.FIELD_FIELD_NUMBER, idx]
                    gi = self._get_doc_import_name

                    field_infos[field.name] = {
                        **args,
                        'comment': self._get_comments(field_scl),
                        'field_type': self._fix_meta_import(gi(field_type),
                                                            mapping={
                                                                'proto.': 'proto.primitives.ProtoType.'
                                                            }),
                        'field_class': self._fix_meta_import(gi(field_class),
                                                             mapping={'proto.': 'proto.fields.'}
                                                             ),
                    }

                docstring = self._generate_docstring_comment_(
                    header=class_comment or [''],
                    field_infos=field_infos,
                    label_prefix=self._get_doc_import_name(class_name)
                )
                self.lines = self.lines[:_line_count] + docstring + self.lines[_line_count:]

            l("\n")

    def protoplus_type(
            self, field: d.FieldDescriptorProto
    ) -> Tuple[Any, str]:
        """
        Generate imports and return correct type
        """

        mapping: Dict[d.FieldDescriptorProto.Type.V, Callable[[], str]] = {
            d.FieldDescriptorProto.TYPE_DOUBLE: lambda: self._import('proto', 'DOUBLE'),
            d.FieldDescriptorProto.TYPE_FLOAT: lambda: self._import('proto', 'FLOAT'),
            d.FieldDescriptorProto.TYPE_INT64: lambda: self._import('proto', 'INT64'),
            d.FieldDescriptorProto.TYPE_UINT64: lambda: self._import('proto', 'UINT64'),
            d.FieldDescriptorProto.TYPE_FIXED64: lambda: self._import('proto', 'FIXED64'),
            d.FieldDescriptorProto.TYPE_SFIXED64: lambda: self._import('proto', 'SFIXED64'),
            d.FieldDescriptorProto.TYPE_SINT64: lambda: self._import('proto', 'SINT64'),
            d.FieldDescriptorProto.TYPE_INT32: lambda: self._import('proto', 'INT32'),
            d.FieldDescriptorProto.TYPE_UINT32: lambda: self._import('proto', 'UINT32'),
            d.FieldDescriptorProto.TYPE_FIXED32: lambda: self._import('proto', 'FIXED32'),
            d.FieldDescriptorProto.TYPE_SFIXED32: lambda: self._import('proto', 'SFIXED32'),
            d.FieldDescriptorProto.TYPE_SINT32: lambda: self._import('proto', 'SINT32'),
            d.FieldDescriptorProto.TYPE_BOOL: lambda: self._import('proto', 'BOOL'),
            d.FieldDescriptorProto.TYPE_STRING: lambda: self._import('proto', 'STRING'),
            d.FieldDescriptorProto.TYPE_BYTES: lambda: self._import('proto', 'BYTES'),
            d.FieldDescriptorProto.TYPE_ENUM: lambda: self._import_message(field.type_name),
            d.FieldDescriptorProto.TYPE_MESSAGE: lambda: self._import_message(field.type_name),
            d.FieldDescriptorProto.TYPE_GROUP: lambda: self._import_message(field.type_name),
        }

        assert field.type in mapping, "Unrecognized type: " + repr(field.type)
        field_type = mapping[field.type]()

        # For non-repeated fields, use normal Field!
        if field.label != d.FieldDescriptorProto.LABEL_REPEATED:
            field_class = self._import('proto', 'Field')
        else:
            # else use MapField or RepeatedField
            msg = self.descriptors.messages.get(field.type_name)
            field_class = (
                self._import('proto', 'MapField')
                if msg is not None and msg.options.map_entry else
                self._import('proto', 'RepeatedField')
            )

        return field_class, field_type

    def write(self) -> str:
        for reexport_idx in self.fd.public_dependency:
            reexport_file = self.fd.dependency[reexport_idx]
            reexport_fd = self.descriptors.files[reexport_file]
            reexport_imp = (
                    reexport_file[:-6].replace("-", "_").replace("/", ".") + "_pb_plus"
            )
            names = (
                    [m.name for m in reexport_fd.message_type]
                    + [m.name for m in reexport_fd.enum_type]
                    + [v.name for m in reexport_fd.enum_type for v in m.value]
                    + [m.name for m in reexport_fd.extension]
            )
            if reexport_fd.options.py_generic_services:
                names.extend(m.name for m in reexport_fd.service)

            if names:
                # n,n to force a reexport (from x import y as y)
                self.from_imports[reexport_imp].update((n, n) for n in names)

        return super().write()


def generate_proto_plus(
        descriptors: Descriptors,
        response: plugin_pb2.CodeGeneratorResponse,
        **kwargs
) -> None:
    generate_inits = kwargs.pop('generate_inits', False)
    readable_imports = kwargs.pop('readable_imports', False)
    quiet = kwargs.pop('quiet', False)

    if generate_inits and not readable_imports:
        warnings.warn(f"Generating proto plus code with 'from ... import ...' statements, to unwrap circular import"
                      f" loops when importing names in __init__.py files since 'generate_inits' was supplied.")

        kwargs['readable_imports'] = True

    for name, fd in descriptors.to_generate.items():
        pkg_writer = PkgWriter(fd=fd, descriptors=descriptors, **kwargs)
        pkg_writer.write_module_attributes()
        pkg_writer.write_enums(
            fd.enum_type, [d.FileDescriptorProto.ENUM_TYPE_FIELD_NUMBER]
        )
        pkg_writer.write_messages(
            fd.message_type, [d.FileDescriptorProto.MESSAGE_TYPE_FIELD_NUMBER]
        )

        if fd.options.py_generic_services:
            sys.stderr.write("GRPC compilation not implemented.")
            sys.exit(1)

        assert name == fd.name
        assert fd.name.endswith(".proto")
        output = response.file.add()
        output.name = fd.name[:-6].replace("-", "_").replace(".", "/") + "_pb_plus.py"
        output.content = HEADER + pkg_writer.write()
        if not quiet:
            print("Writing protoplus to", output.name, file=sys.stderr)

    if not generate_inits:
        return

    # imports need "readable imports" setting
    kwargs['readable_imports'] = True
    root_package = kwargs.pop('package', None)

    # generate init files for packages
    by_package = {k: list(v) for k, v in groupby(descriptors.to_generate.values(), key=lambda d: d.package)}

    for pkg, modules in by_package.items():
        output = response.file.add()
        output.name = pkg.replace('-', '_').replace('.', '/') + "/__init__.py"

        init_writer = InitWriter(
            descriptors=descriptors,
            modules={pkg: modules},
            package=pkg,
            **kwargs
        )
        init_writer.write_module_attributes()
        output.content = INIT_HEADER + init_writer.write()

    root_dir = op.commonprefix([g.name for g in descriptors.to_generate.values()])
    if not root_dir:
        return

    root_init = root_dir.replace('-', '_').replace('.', '/') + '/__init__.py'

    if not any(o.name == root_init for o in response.file):
        output = response.file.add()
        output.name = root_init
        root_writer = InitWriter(
            descriptors=descriptors,
            modules=by_package,
            package=root_package,
            **kwargs
        )
        root_writer.write_module_attributes()
        output.content = INIT_HEADER + root_writer.write()


class InitWriter(Writer):

    def __init__(self,
                 *,
                 descriptors: Descriptors,
                 modules: Dict[str, List[d.FileDescriptorProto]],
                 **kwargs
                 ) -> None:
        super().__init__(fd=d.FileDescriptorProto(), descriptors=descriptors, **kwargs)
        self.modules = modules

    def write_module_attributes(self) -> None:
        l = self._write_line

        l("__all__ = (")
        with self._indent():
            for pkg, modules in self.modules.items():
                for message in chain(chain.from_iterable(m.enum_type for m in modules),
                                     chain.from_iterable(m.message_type for m in modules)):
                    name = (
                        message.name if message.name not in PYTHON_RESERVED else "_r_" + message.name
                    )
                    message = self._import_message(f".{pkg}.{name}")
                    l(f'"{message}",')
        l(")")

    def write(self) -> str:
        def remove_module(package):
            matching = [n for n in self.modules if package.startswith(n)]
            matching = sorted(matching, reverse=True)
            prefix = matching[0] if matching else ''
            return removeprefix(package, prefix)

        key_fixer = (
            functools.partial(removeprefix, prefix=self.package)
            if self.package else
            remove_module
        )

        if self.relative_imports:
            self.from_imports = {key_fixer(k): v for k, v in self.from_imports.items()}
            self.imports = set((key_fixer(imp), alias) for imp, alias in self.imports)

        return super().write()


@contextlib.contextmanager
def code_generation() -> Iterator[
    Tuple[plugin_pb2.CodeGeneratorRequest, plugin_pb2.CodeGeneratorResponse],
]:
    if len(sys.argv) > 1 and sys.argv[1] in ("-V", "--version"):
        print("codestare-proto-plus " + __version__)
        sys.exit(0)

    # Read request message from stdin
    data = sys.stdin.buffer.read()

    # Parse request
    request = plugin_pb2.CodeGeneratorRequest()

    # allow input as json for debug
    try:
        request.ParseFromString(data)
    except DecodeError as e1:
        from google.protobuf.json_format import Parse, ParseError
        try:
            Parse(data, request)
        except ParseError as e2:
            raise e2 from e1

    # Create response
    response = plugin_pb2.CodeGeneratorResponse()

    # Declare support for optional proto3 fields
    response.supported_features |= (
        plugin_pb2.CodeGeneratorResponse.FEATURE_PROTO3_OPTIONAL
    )

    yield request, response

    # Serialise response message
    output = response.SerializeToString()

    # Write to stdout
    sys.stdout.buffer.write(output)


def main() -> None:
    # Generate protoplus
    with code_generation() as (request, response):

        def _get_keyword_param(keyword, parameters):
            found = re.findall(r'{}=(.+?,|.+$)'.format(keyword), parameters)
            if len(found) > 1:
                sys.stderr.write(f"Can't provide {keyword}= multiple times")
                sys.exit(1)

            if not found:
                return None

            return found[0].strip(',')

        kwargs = {key: _get_keyword_param(key, request.parameter) for key in PARAMETER_KEYWORD_ARGS}
        flags = {key: key in request.parameter for key in PARAMETER_FLAGS}

        save_request_path = kwargs.pop('save_request', None)
        if save_request_path:
            file_path = pathlib.Path(save_request_path)
            import google.protobuf.json_format
            with file_path.open('wb') as binary:
                binary.write(request.SerializeToString(deterministic=True))
            with (file_path.parent / f"{file_path.stem}.json").open('w') as json:
                json.write(google.protobuf.json_format.MessageToJson(request))

        generate_proto_plus(
            Descriptors(request),
            response,
            **flags,
            **kwargs
        )


if __name__ == "__main__":
    main()
