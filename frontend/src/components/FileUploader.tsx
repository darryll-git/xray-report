"use client";
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, FileImage } from 'lucide-react';
import { motion } from 'framer-motion';

export default function FileUploader({ onFileSelect }: { onFileSelect: (file: File) => void }) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles?.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.dcm']
    },
    maxFiles: 1
  });

  return (
    <div
      {...getRootProps()}
      className={`relative w-full p-8 border-2 border-dashed rounded-2xl transition-all cursor-pointer flex flex-col items-center justify-center overflow-hidden min-h-[250px] shadow-sm
        ${isDragActive ? 'border-blue-600 bg-blue-50' : 'border-slate-300 bg-white hover:bg-slate-50 hover:border-blue-400'}`}
    >
      <input {...getInputProps()} />
      <motion.div 
        animate={{ y: isDragActive ? -10 : 0 }}
        className="z-10 flex flex-col items-center"
      >
        {acceptedFiles.length > 0 ? (
          <div className="flex flex-col items-center text-slate-800">
            <FileImage className="w-16 h-16 mb-4 text-blue-600" />
            <span className="font-semibold text-lg">{acceptedFiles[0].name}</span>
            <span className="text-sm text-slate-500 mt-1">Click or drag to replace</span>
          </div>
        ) : (
          <>
            <UploadCloud className={`w-16 h-16 mb-4 ${isDragActive ? 'text-blue-500' : 'text-slate-400'}`} />
            <p className="text-lg font-medium text-slate-700">
              {isDragActive ? "Drop the X-Ray here" : "Drag & Drop your Chest X-Ray"}
            </p>
            <p className="text-sm text-slate-500 mt-2">
              Supports PNG, JPG, JPEG, DCM
            </p>
          </>
        )}
      </motion.div>
    </div>
  );
}
