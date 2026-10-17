{{- define "heart-risk-api.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "heart-risk-api.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "heart-risk-api.labels" -}}
app.kubernetes.io/name: {{ include "heart-risk-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "heart-risk-api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "heart-risk-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
