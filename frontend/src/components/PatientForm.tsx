"use client";
import React from 'react';

export default function PatientForm({ formData, setFormData }: { formData: any, setFormData: any }) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm h-full">
      <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
        <span className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">👤</span>
        Patient Information
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-slate-500 mb-1 font-medium">Name</label>
          <input name="name" value={formData.name} onChange={handleChange} className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-slate-800 focus:outline-none focus:border-blue-600 transition-colors" placeholder="Patient's full name" />
        </div>
        <div>
          <label className="block text-sm text-slate-500 mb-1 font-medium">MRN</label>
          <input name="mrn" value={formData.mrn} onChange={handleChange} className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-slate-800 focus:outline-none focus:border-blue-600 transition-colors" placeholder="Medical Record Number" />
        </div>
        <div>
          <label className="block text-sm text-slate-500 mb-1 font-medium">Date of Birth</label>
          <input name="dob" type="date" value={formData.dob} onChange={handleChange} className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-slate-800 focus:outline-none focus:border-blue-600 transition-colors" />
        </div>
        <div>
          <label className="block text-sm text-slate-500 mb-1 font-medium">Gender</label>
          <select name="gender" value={formData.gender} onChange={handleChange} className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-slate-800 focus:outline-none focus:border-blue-600 transition-colors">
            <option>Male</option>
            <option>Female</option>
            <option>Other</option>
          </select>
        </div>
        <div className="md:col-span-2">
          <label className="block text-sm text-slate-500 mb-1 font-medium">Referring Physician</label>
          <input name="referring_physician" value={formData.referring_physician} onChange={handleChange} className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-slate-800 focus:outline-none focus:border-blue-600 transition-colors" placeholder="Physician's name" />
        </div>
        <div className="md:col-span-2">
          <label className="block text-sm text-slate-500 mb-1 font-medium">Clinical History</label>
          <textarea name="history" value={formData.history} onChange={handleChange} rows={2} className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-slate-800 focus:outline-none focus:border-blue-600 transition-colors" placeholder="Routine examination..." />
        </div>
      </div>
    </div>
  );
}
