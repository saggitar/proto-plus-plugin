Plugin to generate proto-plus classes with protoc compiler.

Available plugin parameters [flags]:

* readable_imports  — import in the style "from ... import ..."
* relative_imports  — use relative imports instead of absolute
* generate_inits    — generate __init__.py files, always uses readable_imports to avoid circular imports
* quiet             — don't print compiler output

Available plugin parameters [keywords]:

* save_request — saves input as .json and binary file with specified name
* package — use this string as name for the __package__ module attributes

Example::

    protoc --proto-plus_out=generate_inits,relative_imports,readable_imports,package=mypackage,save_request=test_request:./output_dir schema.proto

