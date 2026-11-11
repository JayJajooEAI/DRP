# assets: what a client would supply

```
assets/
├── <client>/            one folder per client (25), named like 1_1_Ionos, AT_T, PayPal
│   ├── avatar.png       the client's logo; the reference for logo matching
│   └── wordmark.png     the brand's wordmark (shown in the banner of some generated pages)
└── faces/               265 files: 264 AI-generated portraits (male_NNN.jpg, female_NNN.jpg) and map.json
```

`faces/map.json` maps each protected executive's name to the portrait file used for that person. The executives are fictional and the faces are AI-generated, so there is no real person behind any of them.

The same logos and executive photos are also stored in RDS (`assets` table), with their CLIP vectors (`impersonation_embeddings`).
