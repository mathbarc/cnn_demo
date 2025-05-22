all:
	docker build . -t obj_detection
	docker tag obj_detection:latest docker-registry.solv.local/obj_detection:latest
push:
	docker push docker-registry.solv.local/obj_detection:latest
submit:
	argo submit argo_template.yaml
