---
description: 清华开源的on-device推理框架
---

# 🥹 ktransformers

ktransformers 在大部分情况下面都是魔改了llama.cpp做offloading，通过把attention和shared expert放到GPU上面实现了671B满血Deepseek V3推理，对MoE部分和attention部分分别设计了不同的算子从而提升了efficiency，官方的做法使用这种offloading操作对仅仅只用llama.cpp部署的deepseekV3强了9.44x（prefill）与3.03x（decoding）

直接上手ktransformers难度较高，需要先搞明白模型的注入规则以及一些特性的启用，比如chunked prefill，layer-wise prefill 以及marlin算子，官方给了一个能够实现专家并行的注入规则，我们可以根据他提供的yaml模板进行替换

下面首先讲讲后端逻辑，看看ktransformers是怎么通过fastapi将模型的推理打包成服务的

```
// main.py

def create_app():
    cfg = Config()
    app = FastAPI()
    if Config().web_cross_domain:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app)
    if cfg.mount_web:
        mount_index_routes(app)
    return app


def main():
    cfg = Config()

    arg_parser = ArgumentParser(cfg)

    # 初始化消息
    args = arg_parser.parse_args()
    app = create_app()
    custom_openapi(app)
    create_interface(config=cfg, default_args=cfg)
    run_api(
        app=app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()

```

这个是所有app的构建逻辑，首先是使用fastapi创建一个app，所用app的参数可以在config.yaml文件中找到，在构建完成之后，启动mount\_router会挂载一个路由，路由到所有的post和get方法，这里的router经过文件的层层路由，包含了很多method，这里我们只考虑chat的router，看下后端是怎么调用模型的

```
// endpoints/char.py


@router.post('/chat/completions', tags=['openai'])
async def chat_completion(request:Request,create:ChatCompletionCreate):
    id = str(uuid4())

    interface: BackendInterfaceBase = get_interface()
    # input_ids = interface.format_and_tokenize_input_ids(id,messages=create.get_tokenizer_messages())

    input_message = [json.loads(m.model_dump_json()) for m in create.messages]

    if Config().api_key != '':
        assert request.headers.get('Authorization', '').split()[-1] == Config().api_key

    if create.stream:
        from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
        
        async def inner():
            chunk = ChatCompletionChunk(
                id = id,
                choices = [],
                object = 'chat.completion.chunk',
                created = int(time()),
                model = Config().model_name,
            )
            
            async for res in interface.inference(input_message,id, create.temperature, create.top_p):
                if isinstance(res, RawUsage):
                    # at the end of inference, interface.inference() will return the usage of inference
                    raw_usage = res
                    chunk.choices = []
                    chunk.usage = CompletionUsage(
                        prompt_tokens = raw_usage.prefill_count,
                        completion_tokens = raw_usage.decode_count,
                        total_tokens = raw_usage.prefill_count + raw_usage.decode_count
                    )

                    yield chunk

                else:
                    token, finish_reason = res
                    choice = Choice(
                        index = 0,
                        delta = ChoiceDelta(content=token, role=None, tool_calls=None),
                        finish_reason = finish_reason,
                        logprobs = None,
                    )
                    chunk.choices = [choice]
                    yield chunk

        return chat_stream_response(request, inner())
    else:
        from openai.types.chat.chat_completion import Choice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage

        content = ""
        finish_reason = None
        async for res in interface.inference(input_message,id,create.temperature,create.top_p):
            if isinstance(res, RawUsage):
                raw_usage = res
                usage = CompletionUsage(
                    prompt_tokens = raw_usage.prefill_count,
                    completion_tokens = raw_usage.decode_count,
                    total_tokens = raw_usage.prefill_count + raw_usage.decode_count
                )
            else:
                token, finish_reason = res
                content = content + token
                finish_reason = finish_reason

        choice = Choice(
            index = 0,
            finish_reason = finish_reason,
            message = ChatCompletionMessage(
                content=content,
                role="assistant"
            ))

        chat_completion = ChatCompletion(
            id = id,
            choices = [choice],
            created = int(time()),
            model = Config().model_name,
            object = 'chat.completion',
            usage = usage
        )

        return chat_completion

```

这里的chat方法首先使用get\_interface得到后端接口，然后使用interface的inference mode 进行模型推理，最后得到一个chunk是为模型的输出（一个token）。这里大概就将后端调用模型的部分讲的比较清楚了，接下来我们来看下这个interface是如何load模型同时进行推理的。

```
// Some code
class KDeepseekV3MoE(BaseInjectedModule, DeepseekV3MoE):
    
    def forward(self, hidden_states,layer_idx):
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        # recorder.record(topk_idx,topk_weight,layer_idx)
        # recorder.save(layer_idx)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        
        # only for generate phase
        if sequence_length == 1 and hasattr(self.experts.generate_experts, "submit_for_one_decode") and torch.cuda.is_current_stream_capturing():
            self.experts.generate_experts.submit_for_one_decode(hidden_states[0], topk_idx[0], topk_weight[0])
            if self.config.n_shared_experts is not None:
                y_ = self.shared_experts(identity,layer_idx).squeeze(0)
            y = self.experts.generate_experts.sync_for_one_decode().unsqueeze(0)
            y += y_
            y.resize_(*orig_shape)
            return y

        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity,layer_idx).squeeze(0)
            
        if isinstance(self.experts, KExpertsBase):
            y = self.moe_kexperts(hidden_states, topk_idx, topk_weight).view(*orig_shape).to(device=hidden_states.device)
        elif hidden_states.size(0) > 10:
            # TODO may bugs here
            y = (
                self.moe_infer(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        else:
            # TODO may bugs here
            y = (
                self.moe_infer_simple(hidden_states, topk_idx, topk_weight)
                .view(*orig_shape)
                .to(device=hidden_states.device)
            )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def moe_kexperts(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer_simple(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [num_tokens, hidden_size]
        topk_ids, topk_weight: [num_tokens, num_selected_experts]
        """
        outs = torch.zeros_like(x)
        for token_idx in range(topk_ids.size(0)):
            for expert_idx in range(topk_ids.size(1)):
                expert = self.experts[topk_ids[token_idx, expert_idx]]
                outs[token_idx] += (
                    expert.forward(x[token_idx]) * topk_weight[token_idx, expert_idx]
                )
        return outs

    @torch.no_grad()
    # TODO may bugs here
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert.forward(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out
```

