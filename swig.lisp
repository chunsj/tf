;;; This file was automatically generated by SWIG (http://www.swig.org).
;;; Version 3.0.12
;;;
;;; Do not make changes to this file unless you know what you are doing--modify
;;; the SWIG interface file instead.

(cl:defpackage :tf
  (:use #:common-lisp
        #:cffi))

(in-package :tf)

(defparameter *tensorflow-lib* "/usr/local/lib/libtensorflow.dylib")
(defvar *tensorflow-loaded* nil)

(unless *tensorflow-loaded*
  (progn
    (cffi:load-foreign-library *tensorflow-lib*)
    (setf *tensorflow-loaded* T)))

;;;SWIG wrapper code starts here

(cl:defmacro defanonenum (cl:&body enums)
  "Converts anonymous enums to defconstants."
  `(cl:progn ,@(cl:loop for value in enums
                        for index = 0 then (cl:1+ index)
                        when (cl:listp value) do (cl:setf index (cl:second value)
                                                          value (cl:first value))
                        collect `(cl:defconstant ,value ,index))))

(cl:eval-when (:compile-toplevel :load-toplevel)
  (cl:unless (cl:fboundp 'swig-lispify)
    (cl:defun swig-lispify (name flag cl:&optional (package cl:*package*))
      (cl:labels ((helper (lst last rest cl:&aux (c (cl:car lst)))
                    (cl:cond
                      ((cl:null lst)
                       rest)
                      ((cl:upper-case-p c)
                       (helper (cl:cdr lst) 'upper
                               (cl:case last
                                 ((lower digit) (cl:list* c #\- rest))
                                 (cl:t (cl:cons c rest)))))
                      ((cl:lower-case-p c)
                       (helper (cl:cdr lst) 'lower (cl:cons (cl:char-upcase c) rest)))
                      ((cl:digit-char-p c)
                       (helper (cl:cdr lst) 'digit
                               (cl:case last
                                 ((upper lower) (cl:list* c #\- rest))
                                 (cl:t (cl:cons c rest)))))
                      ((cl:char-equal c #\_)
                       (helper (cl:cdr lst) '_ (cl:cons #\- rest)))
                      (cl:t
                       (cl:error "Invalid character: ~A" c)))))
        (cl:let ((fix (cl:case flag
                        ((constant enumvalue) "+")
                        (variable "*")
                        (cl:t ""))))
          (cl:intern
           (cl:concatenate
            'cl:string
            fix
            (cl:nreverse (helper (cl:concatenate 'cl:list name) cl:nil cl:nil))
            fix)
           package))))))

;;;SWIG wrapper code ends here


(cffi:defcfun ("TF_Version" TF_Version) :string)

(cffi:defcenum TF_DataType
	(:TF_FLOAT #.1)
	(:TF_DOUBLE #.2)
	(:TF_INT32 #.3)
	(:TF_UINT8 #.4)
	(:TF_INT16 #.5)
	(:TF_INT8 #.6)
	(:TF_STRING #.7)
	(:TF_COMPLEX64 #.8)
	(:TF_COMPLEX #.8)
	(:TF_INT64 #.9)
	(:TF_BOOL #.10)
	(:TF_QINT8 #.11)
	(:TF_QUINT8 #.12)
	(:TF_QINT32 #.13)
	(:TF_BFLOAT16 #.14)
	(:TF_QINT16 #.15)
	(:TF_QUINT16 #.16)
	(:TF_UINT16 #.17)
	(:TF_COMPLEX128 #.18)
	(:TF_HALF #.19)
	(:TF_RESOURCE #.20))

(cffi:defcfun ("TF_DataTypeSize" TF_DataTypeSize) :unsigned-long
  (dt TF_DataType))

(cffi:defcenum TF_Code
	(:TF_OK #.0)
	(:TF_CANCELLED #.1)
	(:TF_UNKNOWN #.2)
	(:TF_INVALID_ARGUMENT #.3)
	(:TF_DEADLINE_EXCEEDED #.4)
	(:TF_NOT_FOUND #.5)
	(:TF_ALREADY_EXISTS #.6)
	(:TF_PERMISSION_DENIED #.7)
	(:TF_UNAUTHENTICATED #.16)
	(:TF_RESOURCE_EXHAUSTED #.8)
	(:TF_FAILED_PRECONDITION #.9)
	(:TF_ABORTED #.10)
	(:TF_OUT_OF_RANGE #.11)
	(:TF_UNIMPLEMENTED #.12)
	(:TF_INTERNAL #.13)
	(:TF_UNAVAILABLE #.14)
	(:TF_DATA_LOSS #.15))

(cffi:defcfun ("TF_NewStatus" TF_NewStatus) :pointer)

(cffi:defcfun ("TF_DeleteStatus" TF_DeleteStatus) :void
  (arg0 :pointer))

(cffi:defcfun ("TF_SetStatus" TF_SetStatus) :void
  (s :pointer)
  (code TF_Code)
  (msg :string))

(cffi:defcfun ("TF_GetCode" TF_GetCode) TF_Code
  (s :pointer))

(cffi:defcfun ("TF_Message" TF_Message) :string
  (s :pointer))

(cffi:defcstruct TF_Buffer
	(data :pointer)
	(length :unsigned-long)
	(data_deallocator :pointer))

(cffi:defcfun ("TF_NewBufferFromString" TF_NewBufferFromString) :pointer
  (proto :pointer)
  (proto_len :unsigned-long))

(cffi:defcfun ("TF_NewBuffer" TF_NewBuffer) :pointer)

(cffi:defcfun ("TF_DeleteBuffer" TF_DeleteBuffer) :void
  (arg0 :pointer))

(cffi:defcfun ("TF_GetBuffer" TF_GetBuffer) TF_Buffer
  (buffer :pointer))

(cffi:defcfun ("TF_NewTensor" TF_NewTensor) :pointer
  (arg0 TF_DataType)
  (dims :pointer)
  (num_dims :int)
  (data :pointer)
  (len :unsigned-long)
  (deallocator :pointer)
  (deallocator_arg :pointer))

(cffi:defcfun ("TF_AllocateTensor" TF_AllocateTensor) :pointer
  (arg0 TF_DataType)
  (dims :pointer)
  (num_dims :int)
  (len :unsigned-long))

(cffi:defcfun ("TF_TensorMaybeMove" TF_TensorMaybeMove) :pointer
  (tensor :pointer))

(cffi:defcfun ("TF_DeleteTensor" TF_DeleteTensor) :void
  (arg0 :pointer))

(cffi:defcfun ("TF_TensorType" TF_TensorType) TF_DataType
  (arg0 :pointer))

(cffi:defcfun ("TF_NumDims" TF_NumDims) :int
  (arg0 :pointer))

(cffi:defcfun ("TF_Dim" TF_Dim) :long
  (tensor :pointer)
  (dim_index :int))

(cffi:defcfun ("TF_TensorByteSize" TF_TensorByteSize) :unsigned-long
  (arg0 :pointer))

(cffi:defcfun ("TF_TensorData" TF_TensorData) :pointer
  (arg0 :pointer))

(cffi:defcfun ("TF_StringEncode" TF_StringEncode) :unsigned-long
  (src :string)
  (src_len :unsigned-long)
  (dst :string)
  (dst_len :unsigned-long)
  (status :pointer))

(cffi:defcfun ("TF_StringDecode" TF_StringDecode) :unsigned-long
  (src :string)
  (src_len :unsigned-long)
  (dst :pointer)
  (dst_len :pointer)
  (status :pointer))

(cffi:defcfun ("TF_StringEncodedSize" TF_StringEncodedSize) :unsigned-long
  (len :unsigned-long))

(cffi:defcfun ("TF_NewSessionOptions" TF_NewSessionOptions) :pointer)

(cffi:defcfun ("TF_SetTarget" TF_SetTarget) :void
  (options :pointer)
  (target :string))

(cffi:defcfun ("TF_SetConfig" TF_SetConfig) :void
  (options :pointer)
  (proto :pointer)
  (proto_len :unsigned-long)
  (status :pointer))

(cffi:defcfun ("TF_DeleteSessionOptions" TF_DeleteSessionOptions) :void
  (arg0 :pointer))

(cffi:defcfun ("TF_NewGraph" TF_NewGraph) :pointer)

(cffi:defcfun ("TF_DeleteGraph" TF_DeleteGraph) :void
  (arg0 :pointer))

(cffi:defcstruct TF_Input
	(oper :pointer)
	(index :int))

(cffi:defcstruct TF_Output
	(oper :pointer)
	(index :int))

(cffi:defcfun ("TF_GraphSetTensorShape" TF_GraphSetTensorShape) :void
  (graph :pointer)
  (output TF_Output)
  (dims :pointer)
  (num_dims :int)
  (status :pointer))

(cffi:defcfun ("TF_GraphGetTensorNumDims" TF_GraphGetTensorNumDims) :int
  (graph :pointer)
  (output TF_Output)
  (status :pointer))

(cffi:defcfun ("TF_GraphGetTensorShape" TF_GraphGetTensorShape) :void
  (graph :pointer)
  (output TF_Output)
  (dims :pointer)
  (num_dims :int)
  (status :pointer))

(cffi:defcfun ("TF_NewOperation" TF_NewOperation) :pointer
  (graph :pointer)
  (op_type :string)
  (oper_name :string))

(cffi:defcfun ("TF_SetDevice" TF_SetDevice) :void
  (desc :pointer)
  (device :string))

(cffi:defcfun ("TF_AddInput" TF_AddInput) :void
  (desc :pointer)
  (input TF_Output))

(cffi:defcfun ("TF_AddInputList" TF_AddInputList) :void
  (desc :pointer)
  (inputs :pointer)
  (num_inputs :int))

(cffi:defcfun ("TF_AddControlInput" TF_AddControlInput) :void
  (desc :pointer)
  (input :pointer))

(cffi:defcfun ("TF_ColocateWith" TF_ColocateWith) :void
  (desc :pointer)
  (op :pointer))

(cffi:defcfun ("TF_SetAttrString" TF_SetAttrString) :void
  (desc :pointer)
  (attr_name :string)
  (value :pointer)
  (length :unsigned-long))

(cffi:defcfun ("TF_SetAttrStringList" TF_SetAttrStringList) :void
  (desc :pointer)
  (attr_name :string)
  (values :pointer)
  (lengths :pointer)
  (num_values :int))

(cffi:defcfun ("TF_SetAttrInt" TF_SetAttrInt) :void
  (desc :pointer)
  (attr_name :string)
  (value :long))

(cffi:defcfun ("TF_SetAttrIntList" TF_SetAttrIntList) :void
  (desc :pointer)
  (attr_name :string)
  (values :pointer)
  (num_values :int))

(cffi:defcfun ("TF_SetAttrFloat" TF_SetAttrFloat) :void
  (desc :pointer)
  (attr_name :string)
  (value :float))

(cffi:defcfun ("TF_SetAttrFloatList" TF_SetAttrFloatList) :void
  (desc :pointer)
  (attr_name :string)
  (values :pointer)
  (num_values :int))

(cffi:defcfun ("TF_SetAttrBool" TF_SetAttrBool) :void
  (desc :pointer)
  (attr_name :string)
  (value :unsigned-char))

(cffi:defcfun ("TF_SetAttrBoolList" TF_SetAttrBoolList) :void
  (desc :pointer)
  (attr_name :string)
  (values :pointer)
  (num_values :int))

(cffi:defcfun ("TF_SetAttrType" TF_SetAttrType) :void
  (desc :pointer)
  (attr_name :string)
  (value TF_DataType))

(cffi:defcfun ("TF_SetAttrTypeList" TF_SetAttrTypeList) :void
  (desc :pointer)
  (attr_name :string)
  (values :pointer)
  (num_values :int))

(cffi:defcfun ("TF_SetAttrShape" TF_SetAttrShape) :void
  (desc :pointer)
  (attr_name :string)
  (dims :pointer)
  (num_dims :int))

(cffi:defcfun ("TF_SetAttrShapeList" TF_SetAttrShapeList) :void
  (desc :pointer)
  (attr_name :string)
  (dims :pointer)
  (num_dims :pointer)
  (num_shapes :int))

(cffi:defcfun ("TF_SetAttrTensorShapeProto" TF_SetAttrTensorShapeProto) :void
  (desc :pointer)
  (attr_name :string)
  (proto :pointer)
  (proto_len :unsigned-long)
  (status :pointer))

(cffi:defcfun ("TF_SetAttrTensorShapeProtoList" TF_SetAttrTensorShapeProtoList) :void
  (desc :pointer)
  (attr_name :string)
  (protos :pointer)
  (proto_lens :pointer)
  (num_shapes :int)
  (status :pointer))

(cffi:defcfun ("TF_SetAttrTensor" TF_SetAttrTensor) :void
  (desc :pointer)
  (attr_name :string)
  (value :pointer)
  (status :pointer))

(cffi:defcfun ("TF_SetAttrTensorList" TF_SetAttrTensorList) :void
  (desc :pointer)
  (attr_name :string)
  (values :pointer)
  (num_values :int)
  (status :pointer))

(cffi:defcfun ("TF_SetAttrValueProto" TF_SetAttrValueProto) :void
  (desc :pointer)
  (attr_name :string)
  (proto :pointer)
  (proto_len :unsigned-long)
  (status :pointer))

(cffi:defcfun ("TF_FinishOperation" TF_FinishOperation) :pointer
  (desc :pointer)
  (status :pointer))

(cffi:defcfun ("TF_OperationName" TF_OperationName) :string
  (oper :pointer))

(cffi:defcfun ("TF_OperationOpType" TF_OperationOpType) :string
  (oper :pointer))

(cffi:defcfun ("TF_OperationDevice" TF_OperationDevice) :string
  (oper :pointer))

(cffi:defcfun ("TF_OperationNumOutputs" TF_OperationNumOutputs) :int
  (oper :pointer))

(cffi:defcfun ("TF_OperationOutputType" TF_OperationOutputType) TF_DataType
  (oper_out TF_Output))

(cffi:defcfun ("TF_OperationOutputListLength" TF_OperationOutputListLength) :int
  (oper :pointer)
  (arg_name :string)
  (status :pointer))

(cffi:defcfun ("TF_OperationNumInputs" TF_OperationNumInputs) :int
  (oper :pointer))

(cffi:defcfun ("TF_OperationInputType" TF_OperationInputType) TF_DataType
  (oper_in TF_Input))

(cffi:defcfun ("TF_OperationInputListLength" TF_OperationInputListLength) :int
  (oper :pointer)
  (arg_name :string)
  (status :pointer))

(cffi:defcfun ("TF_OperationInput" TF_OperationInput) TF_Output
  (oper_in TF_Input))

(cffi:defcfun ("TF_OperationOutputNumConsumers" TF_OperationOutputNumConsumers) :int
  (oper_out TF_Output))

(cffi:defcfun ("TF_OperationOutputConsumers" TF_OperationOutputConsumers) :int
  (oper_out TF_Output)
  (consumers :pointer)
  (max_consumers :int))

(cffi:defcfun ("TF_OperationNumControlInputs" TF_OperationNumControlInputs) :int
  (oper :pointer))

(cffi:defcfun ("TF_OperationGetControlInputs" TF_OperationGetControlInputs) :int
  (oper :pointer)
  (control_inputs :pointer)
  (max_control_inputs :int))

(cffi:defcfun ("TF_OperationNumControlOutputs" TF_OperationNumControlOutputs) :int
  (oper :pointer))

(cffi:defcfun ("TF_OperationGetControlOutputs" TF_OperationGetControlOutputs) :int
  (oper :pointer)
  (control_outputs :pointer)
  (max_control_outputs :int))

(cffi:defcenum TF_AttrType
	(:TF_ATTR_STRING #.0)
	(:TF_ATTR_INT #.1)
	(:TF_ATTR_FLOAT #.2)
	(:TF_ATTR_BOOL #.3)
	(:TF_ATTR_TYPE #.4)
	(:TF_ATTR_SHAPE #.5)
	(:TF_ATTR_TENSOR #.6)
	(:TF_ATTR_PLACEHOLDER #.7)
	(:TF_ATTR_FUNC #.8))

(cffi:defcstruct TF_AttrMetadata
	(is_list :unsigned-char)
	(list_size :long)
	(type TF_AttrType)
	(total_size :long))

(cffi:defcfun ("TF_OperationGetAttrMetadata" TF_OperationGetAttrMetadata) TF_AttrMetadata
  (oper :pointer)
  (attr_name :string)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrString" TF_OperationGetAttrString) :void
  (oper :pointer)
  (attr_name :string)
  (value :pointer)
  (max_length :unsigned-long)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrStringList" TF_OperationGetAttrStringList) :void
  (oper :pointer)
  (attr_name :string)
  (values :pointer)
  (lengths :pointer)
  (max_values :int)
  (storage :pointer)
  (storage_size :unsigned-long)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrInt" TF_OperationGetAttrInt) :void
  (oper :pointer)
  (attr_name :string)
  (value :pointer)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrIntList" TF_OperationGetAttrIntList) :void
  (oper :pointer)
  (attr_name :string)
  (values :pointer)
  (max_values :int)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrFloat" TF_OperationGetAttrFloat) :void
  (oper :pointer)
  (attr_name :string)
  (value :pointer)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrFloatList" TF_OperationGetAttrFloatList) :void
  (oper :pointer)
  (attr_name :string)
  (values :pointer)
  (max_values :int)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrBool" TF_OperationGetAttrBool) :void
  (oper :pointer)
  (attr_name :string)
  (value :pointer)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrBoolList" TF_OperationGetAttrBoolList) :void
  (oper :pointer)
  (attr_name :string)
  (values :pointer)
  (max_values :int)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrType" TF_OperationGetAttrType) :void
  (oper :pointer)
  (attr_name :string)
  (value :pointer)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrTypeList" TF_OperationGetAttrTypeList) :void
  (oper :pointer)
  (attr_name :string)
  (values :pointer)
  (max_values :int)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrShape" TF_OperationGetAttrShape) :void
  (oper :pointer)
  (attr_name :string)
  (value :pointer)
  (num_dims :int)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrShapeList" TF_OperationGetAttrShapeList) :void
  (oper :pointer)
  (attr_name :string)
  (dims :pointer)
  (num_dims :pointer)
  (num_shapes :int)
  (storage :pointer)
  (storage_size :int)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrTensorShapeProto" TF_OperationGetAttrTensorShapeProto) :void
  (oper :pointer)
  (attr_name :string)
  (value :pointer)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrTensorShapeProtoList" TF_OperationGetAttrTensorShapeProtoList) :void
  (oper :pointer)
  (attr_name :string)
  (values :pointer)
  (max_values :int)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrTensor" TF_OperationGetAttrTensor) :void
  (oper :pointer)
  (attr_name :string)
  (value :pointer)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrTensorList" TF_OperationGetAttrTensorList) :void
  (oper :pointer)
  (attr_name :string)
  (values :pointer)
  (max_values :int)
  (status :pointer))

(cffi:defcfun ("TF_OperationGetAttrValueProto" TF_OperationGetAttrValueProto) :void
  (oper :pointer)
  (attr_name :string)
  (output_attr_value :pointer)
  (status :pointer))

(cffi:defcfun ("TF_GraphOperationByName" TF_GraphOperationByName) :pointer
  (graph :pointer)
  (oper_name :string))

(cffi:defcfun ("TF_GraphNextOperation" TF_GraphNextOperation) :pointer
  (graph :pointer)
  (pos :pointer))

(cffi:defcfun ("TF_GraphToGraphDef" TF_GraphToGraphDef) :void
  (graph :pointer)
  (output_graph_def :pointer)
  (status :pointer))

(cffi:defcfun ("TF_NewImportGraphDefOptions" TF_NewImportGraphDefOptions) :pointer)

(cffi:defcfun ("TF_DeleteImportGraphDefOptions" TF_DeleteImportGraphDefOptions) :void
  (opts :pointer))

(cffi:defcfun ("TF_ImportGraphDefOptionsSetPrefix" TF_ImportGraphDefOptionsSetPrefix) :void
  (opts :pointer)
  (prefix :string))

(cffi:defcfun ("TF_ImportGraphDefOptionsAddInputMapping" TF_ImportGraphDefOptionsAddInputMapping) :void
  (opts :pointer)
  (src_name :string)
  (src_index :int)
  (dst TF_Output))

(cffi:defcfun ("TF_ImportGraphDefOptionsRemapControlDependency" TF_ImportGraphDefOptionsRemapControlDependency) :void
  (opts :pointer)
  (src_name :string)
  (dst :pointer))

(cffi:defcfun ("TF_ImportGraphDefOptionsAddControlDependency" TF_ImportGraphDefOptionsAddControlDependency) :void
  (opts :pointer)
  (oper :pointer))

(cffi:defcfun ("TF_ImportGraphDefOptionsAddReturnOutput" TF_ImportGraphDefOptionsAddReturnOutput) :void
  (opts :pointer)
  (oper_name :string)
  (index :int))

(cffi:defcfun ("TF_ImportGraphDefOptionsNumReturnOutputs" TF_ImportGraphDefOptionsNumReturnOutputs) :int
  (opts :pointer))

(cffi:defcfun ("TF_GraphImportGraphDefWithReturnOutputs" TF_GraphImportGraphDefWithReturnOutputs) :void
  (graph :pointer)
  (graph_def :pointer)
  (options :pointer)
  (return_outputs :pointer)
  (num_return_outputs :int)
  (status :pointer))

(cffi:defcfun ("TF_GraphImportGraphDef" TF_GraphImportGraphDef) :void
  (graph :pointer)
  (graph_def :pointer)
  (options :pointer)
  (status :pointer))

(cffi:defcfun ("TF_OperationToNodeDef" TF_OperationToNodeDef) :void
  (oper :pointer)
  (output_node_def :pointer)
  (status :pointer))

(cffi:defcstruct TF_WhileParams
	(ninputs :int)
	(cond_graph :pointer)
	(cond_inputs :pointer)
	(cond_output TF_Output)
	(body_graph :pointer)
	(body_inputs :pointer)
	(body_outputs :pointer)
	(name :string))

(cffi:defcfun ("TF_NewWhile" TF_NewWhile) TF_WhileParams
  (g :pointer)
  (inputs :pointer)
  (ninputs :int)
  (status :pointer))

(cffi:defcfun ("TF_FinishWhile" TF_FinishWhile) :void
  (params :pointer)
  (status :pointer)
  (outputs :pointer))

(cffi:defcfun ("TF_AbortWhile" TF_AbortWhile) :void
  (params :pointer))

(cffi:defcfun ("TF_AddGradients" TF_AddGradients) :void
  (g :pointer)
  (y :pointer)
  (ny :int)
  (x :pointer)
  (nx :int)
  (dx :pointer)
  (status :pointer)
  (dy :pointer))

(cffi:defcfun ("TF_NewSession" TF_NewSession) :pointer
  (graph :pointer)
  (opts :pointer)
  (status :pointer))

(cffi:defcfun ("TF_LoadSessionFromSavedModel" TF_LoadSessionFromSavedModel) :pointer
  (session_options :pointer)
  (run_options :pointer)
  (export_dir :string)
  (tags :pointer)
  (tags_len :int)
  (graph :pointer)
  (meta_graph_def :pointer)
  (status :pointer))

(cffi:defcfun ("TF_CloseSession" TF_CloseSession) :void
  (arg0 :pointer)
  (status :pointer))

(cffi:defcfun ("TF_DeleteSession" TF_DeleteSession) :void
  (arg0 :pointer)
  (status :pointer))

(cffi:defcfun ("TF_SessionRun" TF_SessionRun) :void
  (session :pointer)
  (run_options :pointer)
  (inputs :pointer)
  (input_values :pointer)
  (ninputs :int)
  (outputs :pointer)
  (output_values :pointer)
  (noutputs :int)
  (target_opers :pointer)
  (ntargets :int)
  (run_metadata :pointer)
  (arg11 :pointer))

(cffi:defcfun ("TF_SessionPRunSetup" TF_SessionPRunSetup) :void
  (arg0 :pointer)
  (inputs :pointer)
  (ninputs :int)
  (outputs :pointer)
  (noutputs :int)
  (target_opers :pointer)
  (ntargets :int)
  (handle :pointer)
  (arg8 :pointer))

(cffi:defcfun ("TF_SessionPRun" TF_SessionPRun) :void
  (arg0 :pointer)
  (handle :string)
  (inputs :pointer)
  (input_values :pointer)
  (ninputs :int)
  (outputs :pointer)
  (output_values :pointer)
  (noutputs :int)
  (target_opers :pointer)
  (ntargets :int)
  (arg10 :pointer))

(cffi:defcfun ("TF_DeletePRunHandle" TF_DeletePRunHandle) :void
  (handle :string))

(cffi:defcfun ("TF_NewDeprecatedSession" TF_NewDeprecatedSession) :pointer
  (arg0 :pointer)
  (status :pointer))

(cffi:defcfun ("TF_CloseDeprecatedSession" TF_CloseDeprecatedSession) :void
  (arg0 :pointer)
  (status :pointer))

(cffi:defcfun ("TF_DeleteDeprecatedSession" TF_DeleteDeprecatedSession) :void
  (arg0 :pointer)
  (status :pointer))

(cffi:defcfun ("TF_Reset" TF_Reset) :void
  (opt :pointer)
  (containers :pointer)
  (ncontainers :int)
  (status :pointer))

(cffi:defcfun ("TF_ExtendGraph" TF_ExtendGraph) :void
  (arg0 :pointer)
  (proto :pointer)
  (proto_len :unsigned-long)
  (arg3 :pointer))

(cffi:defcfun ("TF_Run" TF_Run) :void
  (arg0 :pointer)
  (run_options :pointer)
  (input_names :pointer)
  (inputs :pointer)
  (ninputs :int)
  (output_names :pointer)
  (outputs :pointer)
  (noutputs :int)
  (target_oper_names :pointer)
  (ntargets :int)
  (run_metadata :pointer)
  (arg11 :pointer))

(cffi:defcfun ("TF_PRunSetup" TF_PRunSetup) :void
  (arg0 :pointer)
  (input_names :pointer)
  (ninputs :int)
  (output_names :pointer)
  (noutputs :int)
  (target_oper_names :pointer)
  (ntargets :int)
  (handle :pointer)
  (arg8 :pointer))

(cffi:defcfun ("TF_PRun" TF_PRun) :void
  (arg0 :pointer)
  (handle :string)
  (input_names :pointer)
  (inputs :pointer)
  (ninputs :int)
  (output_names :pointer)
  (outputs :pointer)
  (noutputs :int)
  (target_oper_names :pointer)
  (ntargets :int)
  (arg10 :pointer))

(cffi:defcfun ("TF_LoadLibrary" TF_LoadLibrary) :pointer
  (library_filename :string)
  (status :pointer))

(cffi:defcfun ("TF_GetOpList" TF_GetOpList) TF_Buffer
  (lib_handle :pointer))

(cffi:defcfun ("TF_DeleteLibraryHandle" TF_DeleteLibraryHandle) :void
  (lib_handle :pointer))

(cffi:defcfun ("TF_GetAllOpList" TF_GetAllOpList) :pointer)
